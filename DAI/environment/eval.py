import os
import sys

sys.path.append(os.getcwd())

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from loguru import logger
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

logger.remove()
logger.add(sys.stderr, level="INFO")

import ray
from gymnasium.envs.registration import register
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

import wandb
from DAI.interfaces import Object, ObjectType
from DAI.simulator.extract import (
    get_current_affecting_light_state,
    get_current_max_speed,
    get_current_speed,
    get_objects,
    has_collided,
)
from DAI.simulator.wrappers import CarlaTrafficLightState
from wandb.integration.sb3 import WandbCallback

# Custom modules
from .carla_env_2 import CarlaEnv2
from .carla_setup import setup_carla

# Initialize wandb
wandb.init(
    project="carla_sac_eval",
    config={
        "perfect": True,
        "world_max_speed": 120,
        "max_objects": 30,
        "relevant_distance": 100,
        "total_timesteps": 100000,
        },
)

# Environment configuration
config = wandb.config

# Create the custom environment
def create_env():
    base_env = CarlaEnv2(config)  # Create the base environment
    env = TimeLimit(base_env, max_episode_steps=1000)  # Wrap with TimeLimit
    return env

env = DummyVecEnv([create_env])
num_stacked_frames = 4
env = VecFrameStack(env, num_stacked_frames)  # Wrap with VecFrameStack

model = SAC.load("/mnt/storage/resultsRL/LeadingCar_cv_perfect2_80000", env=env, verbose=1)
print("loaded: ", model)

obs = env.reset()
i = 0
for _ in range(2000):
    i += 1
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    # print("observation shape: ", obs.shape)
    if np.any(dones):
        obs = env.reset()
    print("eval percentage: ", f"{i}/2000 ", 100 * i / 2000, "%")

# Finish the wandb run
wandb.finish()

# Finish the wandb run
wandb.finish()