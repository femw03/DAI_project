import wandb
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
    base_env = CarlaEnv(env_config)
    # Wrap with TimeLimit to set max_episode_steps
    return TimeLimit(base_env, max_episode_steps=10)

def main():
    logger.info("Starting setup...")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Initialize wandb
    wandb.init(project="carla_sac", config={
        "perfect": True,
        "world_max_speed": 100,
        "max_objects": 30,
        "relevant_distance": 25,
        "total_timesteps": 100000,
    })
    
    # Environment configuration
    config = wandb.config

    # Create the custom environment
    base_env = CarlaEnv(config)  # Create the base environment
    env = TimeLimit(base_env, max_episode_steps=1000)  # Wrap with TimeLimit

    # Access the base environment to wait for Carla initialization
    logger.info("Waiting for Carla world to initialize...")
    while base_env.world.car is None:  # Use base_env here
        pass  # Keep looping until env.world is not None
    
    logger.info("Carla world initialized!")

    # Initialize the SAC agent
    model = SAC("MlpPolicy", env, verbose=1)

    # Training Loop
    total_timesteps = config.total_timesteps  # Use the value from wandb config
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save the trained model
    model.save("sac_custom_env_model")
    wandb.save("sac_custom_env_model.zip")

    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()