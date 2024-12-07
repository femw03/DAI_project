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

    model = SAC("MlpPolicy", env, verbose=1)
    # Load the previously trained model
    #model = SAC.load("sac_OnlySpeedLimit_correctTerminate", env=env, verbose=1)
    #print("loaded: ", model)

    # Define save frequency (e.g., every 10,000 timesteps)
    save_frequency = 25000
    total_timesteps = 100000  # Total timesteps to train
    n_steps = save_frequency  # Steps per save

    for step in range(0, total_timesteps, n_steps):
        model.learn(total_timesteps=n_steps, reset_num_timesteps=False, progress_bar=True, callback=WandbCallback())
        # Save the model after every `save_frequency` timesteps
        model.save(f"sac_OnlySpeed_step_{step + n_steps}")
        wandb.save(f"sac_OnlySpeed_step_{step + n_steps}.zip")
        print(f"Model saved at step: {step + n_steps}")

    # Save the final model
    model.save("sac_OnlySpeed_final")
    wandb.save("sac_OnlySpeed_final.zip")

    # Finish the training wandb run 
    wandb.finish() 
    # Start a new wandb run for evaluation 
    wandb.init(project="carla_sac_eval", config=config)

    obs, _ = env.reset()  # Adjusting for SB3 VecEnv API
    i = 0
    for _ in range(10000):
        i += 1
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, infos = env.step(action)
        
        #env.render()
        if terminated or truncated:
            obs, _ = env.reset()  # Adjusting for SB3 VecEnv API
        print("eval percentage: ", f"{i}/10000 ", 100*i/10000, "%")
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()