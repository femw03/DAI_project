import time
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import TimeLimit
import wandb

# Custom Environment import
from env_full_changing_speed_limit import AdaptiveCruiseControlEnv  # Adjust this path if needed

# Create the environment and wrap it to limit episode length
env = make_vec_env(lambda: TimeLimit(AdaptiveCruiseControlEnv(), max_episode_steps=1000), n_envs=1)
env = VecFrameStack(env, n_stack=4)  # Stack the last 4 frames

# Define SAC configuration
model = SAC(
    "MlpPolicy", env,
    gamma=0.99,
    tau=0.005,
    learning_rate=1e-4,
    batch_size=64,
    ent_coef="auto",
    verbose=1
)

# Load the model
#model = SAC.load("Complex_mock_full_stop6", env=env)

# Train the SAC agent
model.learn(total_timesteps=2000000)

# Save the model
model.save("Complex_fromScratch")
wandb.save("Complex_fromScratch.zip")

# Load the model
model = SAC.load("Complex_fromScratch")

# Create the environment for evaluation (with render mode) 
eval_env = make_vec_env(lambda: TimeLimit(AdaptiveCruiseControlEnv(render_mode="human"), max_episode_steps=1000), n_envs=1)
eval_env = VecFrameStack(eval_env, n_stack=4)  # Stack the last 4 frames for evaluation as well

# Evaluation loop with GUI visualization
obs = env.reset()
done = [False]

# Initialize pygame
pygame.init()

# Create a small window to capture events
window = pygame.display.set_mode((200, 100))
pygame.display.set_caption("Press any key to start evaluation")

# Display message in console
print("Press any key to start evaluation")

# Wait loop for key press or quit
waiting = True
while waiting:
    # Poll events from the queue
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:  # Detect any key press
            waiting = False
        elif event.type == pygame.QUIT:  # Detect window close
            waiting = False

# Close the pygame window after detecting key press or close event
pygame.quit()

print("Starting evaluation...")

episode_reward = 0.0

while not done[0]:
    # Get action from the trained agent's policy
    action, _states = model.predict(obs, deterministic=True)
    
    # Step the environment
    obs, reward, done, info = env.step(action)
    episode_reward += reward[0]
    
    # Render the environment to visualize agent behavior
    env.render()
    
    # Slow down the loop to make the GUI visualization observable
    time.sleep(0.1)

# Print total reward for the evaluation episode
print(f"Evaluation completed. Total reward: {episode_reward}")

# Clean up
env.close()