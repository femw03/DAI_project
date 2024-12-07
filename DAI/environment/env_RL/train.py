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
from wandb.integration.sb3 import WandbCallback

# Custom modules
from ...cv import ComputerVisionModuleImp
from .carlaEnv import CarlaEnv
from .carla_setup import setup_carla

def carla_env_creator(env_config):
    """Environment creator for CarlaEnv."""
    base_env = CarlaEnv(env_config)
    # Wrap with TimeLimit to set max_episode_steps
    return TimeLimit(base_env, max_episode_steps=1000)

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

    # Load the previously trained model
    model = SAC.load("sac_OnlySpeedLimit_correctTerminate", env=env, verbose=1)
    print("loaded: ", model)

    # Continue training the model
    additional_timesteps = 400000  # Set this to the number of additional timesteps you want to train for
    model.learn(total_timesteps=additional_timesteps, progress_bar=True, callback=WandbCallback())

    # Save the updated model
    model.save("sac_KeepingDistance_plus_other_cars")
    wandb.save("sac_KeepingDistance_plus_other_cars.zip")

    # Finish the training wandb run 
    wandb.finish() 
    # Start a new wandb run for evaluation 
    wandb.init(project="carla_sac_eval", config=config)

    obs, _ = env.reset()  # Adjusting for SB3 VecEnv API
    i = 0
    for _ in range(100):
        i += 1
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, infos = env.step(action)
        
        #env.render()
        if terminated or truncated:
            obs, _ = env.reset()  # Adjusting for SB3 VecEnv API
        print("eval percentage: ", f"{i}/100 ", 100*i/100, "%")
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()