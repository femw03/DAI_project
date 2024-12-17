"""Implements the final and real application agent"""

from collections import deque
from typing import Deque

import torch
from stable_baselines3.sac import SAC

from ..interfaces import AgentFeatures, CruiseControlAgent


class SACAgent(CruiseControlAgent):
    def __init__(self, weights: str):
        super().__init__()
        self.observations: Deque[torch.Tensor] = deque(maxlen=4)
        self.model = SAC.load(weights)

    def get_action(self, state: AgentFeatures):
        """First stack 4 observations and return the a reasonable action if 4 frames are collected"""
        self.observations.append(state.to_tensor())
        if len(self.observations) < 4:
            return 0.5
        observation = torch.concat(list(self.observations), dim=0)
        action, _ = self.model.predict(observation, deterministic=True)
        return action[0]
