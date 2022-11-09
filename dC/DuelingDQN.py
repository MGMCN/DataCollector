"""
---
title: Deep Q Network (DQN) Model
summary: Implementation of neural network model for Deep Q Network (DQN).
---
# Deep Q Network (DQN) Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/fe1ad986237511ec86e8b763a2d3f710)
"""

import torch
from torch import nn


# from labml_helpers.module import Module


class DuelingModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # The first convolution layer takes a
            # $6\times6$ frame and produces a $3\times3$ frame
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=3),
            nn.ReLU(),

            # The second convolution layer takes a
            # $3\times3$ frame and produces a $2\times2$ frame
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.ReLU(),

            # The third convolution layer takes a
            # $9\times9$ frame and produces a $7\times7$ frame
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # nn.ReLU(),
        )

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # $4$ features
        self.lin = nn.Linear(in_features=3 * 3 * 1, out_features=9)
        self.activation = nn.ReLU()

        # This head gives the state value $V$
        self.state_value = nn.Sequential(
            nn.Linear(in_features=9, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
        )
        # This head gives the action value $A$
        self.action_value = nn.Sequential(
            nn.Linear(in_features=9, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=3),
        )

    def forward(self, obs: torch.Tensor):
        # Convolution
        h = self.conv(obs)
        # Reshape for linear layers
        h = h.reshape((-1, 3 * 3 * 1))

        # Linear layer
        h = self.activation(self.lin(h))

        # $A$
        action_value = self.action_value(h)
        # $V$
        state_value = self.state_value(h)

        # $A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')$
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        # $Q(s, a) =V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')\Big)$
        q = state_value + action_score_centered

        return q
