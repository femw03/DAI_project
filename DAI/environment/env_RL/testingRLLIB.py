import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from .carlaEnv import CarlaEnv

def carla_env_creator(env_config):
    return CarlaEnv(env_config)

# Register the environment with RLlib
register_env("CarlaEnv", carla_env_creator)

# Initialize Ray
ray.init()

# Configure the PPO trainer
config = {
    "env": "CarlaEnv",  # The environment we created
    "num_gpus": 0,  # Set the number of GPUs to use (0 if no GPU)
    "num_workers": 1,  # Number of parallel workers
    "framework": "torch",  # Use PyTorch as the framework
}

# Create a PPO trainer
trainer = ppo.PPOTrainer(config=config)

# Training loop
for i in range(10):
    result = trainer.train()
    print(f"Iteration {i}: reward {result['episode_reward_mean']}")
