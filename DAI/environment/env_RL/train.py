import sys
import numpy as np
import gymnasium as gym
from loguru import logger
import warnings

logger.remove()
logger.add(sys.stderr, level="INFO")

import ray
from ray.rllib.algorithms.sac import SACConfig
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
from ray.tune.registry import register_env
from stable_baselines3 import SAC

# Custom modules
from ...cv import ComputerVisionModuleImp
from .carlaEnv import CarlaEnv
from .carla_setup import setup_carla

def carla_env_creator(env_config):
    """Environment creator for CarlaEnv."""
    return CarlaEnv(env_config)

def main():
    logger.info("Starting setup...")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Environment configuration
    config = {
        "perfect": True,
        "world_max_speed": 100,
        "max_objects": 30,
        "relevant_distance": 25,
    }

    # Create the custom environment
    env = CarlaEnv(config)

    # Initialize the SAC agent
    model = SAC("MlpPolicy", env, verbose=1)

    logger.info("Waiting for Carla world to initialize...")
    while env.world.car is None:
        pass  # Keep looping until env.world is not None
    
    logger.info("Carla world initialized!")

    # Training Loop
    total_timesteps = 100000  # Define the number of timesteps to train the agent
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("sac_custom_env_model")

    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()