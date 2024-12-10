import os
import sys

sys.path.append(os.getcwd())

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

import ray
import wandb
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback

from DAI.interfaces import Object, ObjectType
from DAI.simulator.extract import (
    get_current_affecting_light_state,
    get_current_max_speed,
    get_current_speed,
    get_objects,
    has_collided,
)
from DAI.simulator.wrappers import CarlaTrafficLightState

# Custom modules
from ...cv import ComputerVisionModuleImp
from .carla_env import CarlaEnv
from .carla_setup import setup_carla

# Initialize wandb
wandb.init(
    project="carla_sac_eval",
    config={
        "perfect": True,
        "world_max_speed": 100,
        "max_objects": 30,
        "relevant_distance": 25,
        "total_timesteps": 400000,
    },
)


# Environment configuration
config = wandb.config

# Create the custom environment
base_env = CarlaEnv(config)  # Create the base environment
env = TimeLimit(base_env, max_episode_steps=1000)  # Wrap with TimeLimit

model = SAC.load("sac_CarsOnly_final", env=env, verbose=1)
print("loaded: ", model)

# Define save frequency
save_frequency = 50000
total_timesteps = 400000  # Total timesteps to train
n_steps = save_frequency  # Steps per save


obs, _ = env.reset()  # Adjusting for SB3 VecEnv API
i = 0
for _ in range(10000):
    i += 1
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, infos = env.step(action)

    # env.render()
    if terminated or truncated:
        obs, _ = env.reset()  # Adjusting for SB3 VecEnv API
    print("eval percentage: ", f"{i}/10000 ", 100 * i / 10000, "%")
# Finish the wandb run
wandb.finish()
