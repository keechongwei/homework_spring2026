"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 256, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.model = nn.Sequential(
            # Simple MLP Architecture with ReLU Activations
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),

            # Output size is chunk_size * action_dim to predict the entire action chunk
            nn.Linear(hidden_dims[2], chunk_size * action_dim),
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        predicted_action_chunk = self.model(state) 
        predicted_action_chunk = predicted_action_chunk.view(-1, self.chunk_size, self.action_dim)
        criterion = nn.MSELoss()
        loss = criterion(predicted_action_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        predicted_action_chunk = self.model(state)
        predicted_action_chunk = predicted_action_chunk.view(-1, self.chunk_size, self.action_dim)
        return predicted_action_chunk[:,:num_steps,:]


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 256, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.model = nn.Sequential(

            # Simple MLP Architecture with ReLU Activations
            # state_dim must be modified to be state_dim + (action_dim * chunk_size) + tau
            nn.Linear(state_dim+ (action_dim * chunk_size) + 1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),

            # Output size is chunk_size * action_dim to predict the entire action chunk
            nn.Linear(hidden_dims[2], chunk_size * action_dim),
        )

    # learn vector field here
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        batch_size = state.shape[0]

        # a_t is of shape (batch_size, chunk_size, action_dim)
        a_t = action_chunk

        # A_(t,0). This initialises a tensor with shape like action_chunk
        # each dimension is filled with a value sampled from a Standard Gaussian of N(0,1)
        # this ensures that noise is i.i.d
        a_0 = torch.randn_like(action_chunk)

        # tau, samples from a uniform distribution in range [0,1). 
        # Arbitrarily picks a time step,A_(t,tau) in the diffusion process between A_(t,0) and A_(t)
        tau = torch.rand(batch_size, 1)

        # reshape tau so it can be broadcasting during interpolation
        tau_broadcast = tau.view(batch_size, 1, 1)

        # interpolate to get A_(t,tau) 
        # where A_(t,tau) = tau * A_t + (1-tau) * A_(t,0)
        a_tau = tau_broadcast * a_t + (1.0 - tau_broadcast) * a_0

        # velocity at point is d(a_tau)/d_tau
        # which equals A_t - A_(t,0)
        target_velocity = a_t - a_0

        # Prepare model input (MLP model expects 2D tensor of shape (batch_size,state_dim))
        a_tau = a_tau.view(batch_size, -1)

        # state_dim = state_dim + chunk_size * batch_size + 1
        model_input = torch.cat([state, a_tau, tau], dim=1)

        # predict velocity
        pred_velocity = self.model(model_input)

        # reshapes pred_velocity to shape of target_velocity so that loss can be computed
        pred_velocity = pred_velocity.view_as(target_velocity)

        criterion = nn.MSELoss()
        loss = criterion(pred_velocity, target_velocity)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        batch_size = state.shape[0]

        # Initialise point by randomly sampling from Standard Gaussian Distribution N(0,I)
        action = torch.randn(batch_size, self.chunk_size, self.action_dim)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            # every sample in batch gets same tau, which represents DENOISING timestep
            # so different actions in batch can be at different actual time but same DENOISING timestep
            # since they are being batch processed
            tau = torch.full((batch_size, 1), i / num_steps)

            action_flat = action.view(batch_size, -1)
            model_input = torch.cat([state, action_flat, tau], dim=1)

            # 2. Predict velocity
            velocity = self.model(model_input)
            velocity = velocity.view_as(action)

            # 3. Euler update
            action = action + dt * velocity

        return action


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
