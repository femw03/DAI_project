import time
import numpy as np
import pygame
import optuna  # Import Optuna for hyperparameter tuning
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from adaptive_cruise_control_env import AdaptiveCruiseControlEnv


# Define the function to create environment for training
def create_train_env():
    return TimeLimit(AdaptiveCruiseControlEnv(), max_episode_steps=1000)

# Define the Optuna objective function for hyperparameter tuning
def objective(trial):
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    n_steps = trial.suggest_int("n_steps", 16, 2048, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, log=True)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-2)
    vf_coef = trial.suggest_uniform("vf_coef", 0.1, 0.5)
    
    # Create the environment for training
    env = make_vec_env(create_train_env, n_envs=1)

    # Define the PPO model with the hyperparameters from Optuna
    model = PPO(
        "MlpPolicy",  # MLP policy
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0  # Set verbose to 0 to reduce logs during training
    )

    # Train the PPO agent
    model.learn(total_timesteps=500000)

    # Evaluate the trained model to return a reward score
    eval_env = make_vec_env(lambda: TimeLimit(AdaptiveCruiseControlEnv(), max_episode_steps=1000), n_envs=1)
    obs = eval_env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]  # Summing the rewards

    eval_env.close()
    return episode_reward  # Return the total reward as the objective

# Create Optuna study to optimize hyperparameters
study = optuna.create_study(direction="maximize")  # Maximize the reward
study.optimize(objective, n_trials=20)

# Print the best hyperparameters found
print("Best hyperparameters found: ", study.best_params)

# Train the final model with the best hyperparameters
best_params = study.best_params
env = make_vec_env(create_train_env, n_envs=1)
model = PPO(
    "MlpPolicy",  # MLP policy
    env,
    learning_rate=best_params["learning_rate"],
    n_steps=best_params["n_steps"],
    batch_size=best_params["batch_size"],
    n_epochs=best_params["n_epochs"],
    gamma=best_params["gamma"],
    ent_coef=best_params["ent_coef"],
    vf_coef=best_params["vf_coef"],
    verbose=1  # Set verbose to 1 to see training info
)

# Train the final PPO agent with the optimized hyperparameters
model.learn(total_timesteps=50000)

# Save the model
model.save("ppo_acc_model_full")

# Load the model
model = PPO.load("ppo_acc_model_full", env=env)

# Create the environment for evaluation (with render mode)
def create_eval_env():
    env = AdaptiveCruiseControlEnv()  # Pass render_mode here
    return TimeLimit(env, max_episode_steps=1000)

# Create the environment for evaluation (with render mode)
eval_env = make_vec_env(create_eval_env, n_envs=1)

# Evaluation loop with GUI visualization
obs = eval_env.reset()
done = False  # Change from list to a single boolean flag

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

# Run the evaluation until the episode ends
while not done:
    # Get action from the trained agent's policy
    action, _states = model.predict(obs, deterministic=True)

    # Step the environment
    obs, reward, done, info = eval_env.step(action)  # Handle 'truncated' as well
    episode_reward += reward[0]

    # Render the environment to visualize agent behavior
    eval_env.render()

    # Slow down the loop to make the GUI visualization observable
    time.sleep(0.1)

# Print total reward for the evaluation episode
print(f"Evaluation completed. Total reward: {episode_reward}")

# Clean up
eval_env.close()
