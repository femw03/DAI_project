import sys
import warnings

import numpy as np
from gymnasium.wrappers import TimeLimit
from loguru import logger
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import wandb
from wandb.integration.sb3 import WandbCallback

# Custom modules
from ..environment import CarlaEnv2

logger.remove()
logger.add(sys.stderr, level="ERROR")


def main():
    logger.info("Starting setup...")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Initialize wandb
    wandb.init(
        project="carla_sac",
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
    base_env = CarlaEnv2(config)  # Create the base environment
    time_limited_env = TimeLimit(
        base_env, max_episode_steps=1000
    )  # Wrap with TimeLimit

    # Vectorize the environment and apply frame stacking
    vec_env = DummyVecEnv([lambda: time_limited_env])
    env = VecFrameStack(vec_env, n_stack=4)

    # Access the base environment to wait for Carla initialization
    logger.info("Waiting for Carla world to initialize...")
    while base_env.world.car is None:
        pass  # Keep looping until env.world is not None

    logger.info("Carla world initialized!")

    #model = SAC("MlpPolicy", env, verbose=1)
    #print("made model: ", model)
    # Load the previously trained model
    model = SAC.load("/mnt/storage/resultsRL/Stop_cv_perfect_5000.zip", env=env, verbose=1)
    print("loaded: ", model)

    # Define save frequency
    save_frequency = 5000
    total_timesteps = 100000  # Total timesteps to train
    n_steps = save_frequency  # Steps per save

    # Initialize WandbCallback
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"{wandb.run.dir}/models/",
        model_save_freq=25000,
        verbose=2,
    )

    for step in range(0, total_timesteps, n_steps):
        model.learn(
            total_timesteps=n_steps,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=wandb_callback,
        )
        # Save the model after every `save_frequency` timesteps
        model.save(f"/mnt/storage/resultsRL/Stop1_cv_perfect_{step + n_steps}")
        wandb.save(
            f"/mnt/storage/resultsRL/Stop1_cv_perfect_{step + n_steps}.zip"
        )
        print(f"Model saved at step: {step + n_steps}")

    # Save the final model
    model.save("Stop_cv_perfect_final")
    wandb.save("Stop_cv_perfect_final.zip")

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
        # print("observation shape: ", obs.shape)
        if np.any(dones):
            obs = env.reset()
        print("eval percentage: ", f"{i}/10000 ", 100 * i / 10000, "%")
    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

# successs!!!!
# DAI/scripts/Saves/sac_NewReward_follow_cars1_50000.zip
# /mnt/storage/resultsRL/Continue_on_mock_10000.zip
# /mnt/storage/resultsRL/New_town2_80000.zip # seems very good!!!

# Minimum requirements!
# /mnt/storage/resultsRL/LeadingCar_cv_perfect2_80000.zip # works very well using cv!!!