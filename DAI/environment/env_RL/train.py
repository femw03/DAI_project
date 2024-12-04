import numpy as np
import gymnasium as gym
from loguru import logger

import ray 
from ray import tune 
from ray.rllib.agents.sac import SACTrainer 
from ray.rllib.env.env_context import EnvContext

from DAI.cv import ComputerVisionModuleImp
from DAI.interfaces import CarlaData
from DAI.simulator import CarlaWorld
from DAI.visuals import Visuals
from carlaEnv import CarlaEnv


def set_view_data(data: CarlaData, visuals: Visuals) -> None:
    visuals.depth_image = data.lidar_data.get_lidar_bytes()
    visuals.rgb_image = data.rgb_image.get_image_bytes()
    
def stop(world: CarlaWorld) -> None:
    global is_running
    world.stop()
    is_running = False

def setup_carla() -> CarlaWorld:
    visuals = Visuals(1280, 720, 30)
    world = CarlaWorld(view_height=visuals.height, view_width=visuals.width)
    world.add_listener(set_view_data)

    visuals.on_quit = stop

    world.start()
    visuals.start()

    world.join()
    visuals.join()


def main():
    world = setup_carla()
    cv = ComputerVisionModuleImp()
    
    # Configuration for the environment
    config = {
        "perfect": True,
        "world_max_speed": 100,
        "world": world,             # instance of Carla world
        "max_objects": 30,
        "relevant_distance": 25,
    }

    # RLlib configuration 
    tune_config = { 
        "env": CarlaEnv,            # Set the environment 
        "env_config": config,       # Pass the configuration to the environment 
        "framework": "torch",       # You can use "tf" for TensorFlow 
        "num_gpus": 0,              # Adjust based on your setup 
        "num_workers": 1,           # Number of parallel workers 
        "train_batch_size": 256, 
        "gamma": 0.99, 
        "tau": 0.005, 
        "actor_lr": 0.0003, 
        "critic_lr": 0.0003, 
        "target_network_update_freq": 1, 
        "Q_model": { 
            "fcnet_hiddens": [256, 256], 
            "fcnet_activation": "relu", 
            }, 
        "policy_model": { 
            "fcnet_hiddens": [256, 256], 
            "fcnet_activation": "relu", 
            }, 
        }

    # Initialize Ray 
    ray.init(ignore_reinit_error=True) 
    
    # Run the training 
    tune.run(SACTrainer, config=tune_config, stop={"timesteps_total": 100000}, local_dir="./sac_carla_env") 
    
    # Shutdown Ray 
    ray.shutdown()

if __name__ == "__main__":
    main()
