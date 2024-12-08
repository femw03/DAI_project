import wandb
import sys
import numpy as np
import gymnasium as gym
from loguru import logger
import warnings
from collections import deque

logger.remove()
logger.add(sys.stderr, level="ERROR")

import ray
from ray.rllib.algorithms.sac import SACConfig
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register
from ray.tune.registry import register_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from wandb.integration.sb3 import WandbCallback

# Custom modules
from ...cv import ComputerVisionModuleImp
from .carlaEnv import CarlaEnv
from .carla_setup import setup_carla

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
    time_limited_env = TimeLimit(base_env, max_episode_steps=1000)  # Wrap with TimeLimit

    # Vectorize the environment and apply frame stacking
    vec_env = DummyVecEnv([lambda: time_limited_env])
    env = VecFrameStack(vec_env, n_stack=4)

    # Access the base environment to wait for Carla initialization
    logger.info("Waiting for Carla world to initialize...")
    while base_env.world.car is None:
        pass  # Keep looping until env.world is not None
    
    logger.info("Carla world initialized!")

    model = SAC("MlpPolicy", env, verbose=1)
    print("made model: ", model)
    # Load the previously trained model
    #model = SAC.load("sac_CarsOnlyBusy_25000", env=env, verbose=1)
    #print("loaded: ", model)

    # Define save frequency
    save_frequency = 25000
    total_timesteps = 100000  # Total timesteps to train
    n_steps = save_frequency  # Steps per save

    # Initialize WandbCallback
    wandb_callback = WandbCallback(gradient_save_freq=1000, model_save_path=f"{wandb.run.dir}/models/", model_save_freq=25000, verbose=2)

    for step in range(0, total_timesteps, n_steps):
        model.learn(total_timesteps=n_steps, reset_num_timesteps=False, progress_bar=True, callback=wandb_callback)
        # Save the model after every `save_frequency` timesteps
        model.save(f"/mnt/storage/resultsRL/sac_OnlySpeedFrameStack_{step + n_steps}")
        wandb.save(f"/mnt/storage/resultsRL/sac_OnlySpeedFrameStack_{step + n_steps}.zip")
        print(f"Model saved at step: {step + n_steps}")

    # Save the final model
    model.save("sac_OnlySpeedFrameStack_final")
    wandb.save("sac_OnlySpeedFrameStack_final.zip")

    # Finish the training wandb run 
    wandb.finish() 
    # Start a new wandb run for evaluation 
    wandb.init(project="carla_sac_eval", config=config)

    obs = env.reset()
    i = 0
    for _ in range(10000):
        i += 1
        action, _states = model.predict(obs)
        obs, rewards, dones, infos = env.step(action)
        #print("observation shape: ", obs.shape)
        if np.any(dones):
            obs = env.reset()
        print("eval percentage: ", f"{i}/10000 ", 100*i/10000, "%")
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()