import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        # output of network is [batch,1]
        # want to remove last dim so its shape is [batch] which matches other arrays like q_values
        return self.network(obs).squeeze(-1)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # output from MLP will be [batch,1]
        predicted_q_values = self.forward(obs)

        # TODO: compute the loss using the observations and q_values
        # Assume MSE is used
        loss = F.mse_loss(predicted_q_values,q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # TODO: perform an optimizer step
        
        self.optimizer.step()

        return {
            "Baseline Loss": loss.item(),
        }