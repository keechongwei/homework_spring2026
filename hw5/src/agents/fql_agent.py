from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        with torch.no_grad():
            ac_dim = self.action_dim
            z = torch.randn(
                (1, self.action_dim),
                device=observation.device,
                dtype=observation.dtype,
            )

            action = self.onestep_actor(observation, z)
        
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        dt = 1.0 / self.flow_steps
        action = noise

        for k in range(self.flow_steps):
            t = torch.full(
                (observation.shape[0], 1),
                k / self.flow_steps,
                device=observation.device,
                dtype=observation.dtype,
            )
            velocity = self.bc_actor(observation, action, t)
            action = action + dt * velocity

        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        q = self.critic(observations, actions)
        z = torch.randn((observations.shape[0], self.action_dim), device=observations.device, dtype=observations.dtype)
        with torch.no_grad():
            next_actions = self.onestep_actor(next_observations,z)
            next_actions = torch.clamp(next_actions, -1, 1)

            target_q = self.target_critic(next_observations, next_actions)  
            target_q = target_q.mean(dim=0)  

            targets = rewards + self.discount * (1 - dones) * target_q  
        loss = ((q - targets.unsqueeze(0)) ** 2).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        t = torch.rand((observations.shape[0], 1), device=observations.device, dtype=observations.dtype)
        z = torch.randn((observations.shape[0], self.action_dim), device=observations.device, dtype=observations.dtype)
        a_tilde = (1-t)*z + t*actions
        v = self.bc_actor(observations, a_tilde, t)

        loss =  ((v - (actions-z)) ** 2).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        z = torch.randn((observations.shape[0], self.action_dim),
                        device=observations.device,
                        dtype=observations.dtype)
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        onestep_action = self.onestep_actor(observations,z)
        with torch.no_grad():
            bc_action = self.get_bc_action(observations, z)
        distill_loss = onestep_action - bc_action
        distill_loss = self.alpha * (distill_loss ** 2).mean()

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        q_loss = -self.critic(observations, torch.clamp(onestep_action, -1, 1)).mean()


        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = (actions - onestep_action).pow(2).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)

