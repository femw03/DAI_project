import time
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
import wandb
import optuna
from adaptive_cruise_control_env import AdaptiveCruiseControlEnv

def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    print("Starting a new trial with Optuna...")  # Notify start of a new trial

    # Suggest hyperparameters
    gamma = trial.suggest_float('gamma', 0.95, 0.9999, log=True)
    tau = trial.suggest_float('tau', 1e-4, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.1, 0.01, 0.001])

    print(f"Hyperparameters for this trial: gamma={gamma}, tau={tau}, learning_rate={learning_rate}, batch_size={batch_size}, ent_coef={ent_coef}")

    # Create training environment
    env = make_vec_env(lambda: TimeLimit(AdaptiveCruiseControlEnv(), max_episode_steps=1000), n_envs=1)

    # Initialize SAC model
    model = SAC(
        "MlpPolicy", env,
        gamma=gamma,
        tau=tau,
        learning_rate=learning_rate,
        batch_size=batch_size,
        ent_coef=ent_coef,
        verbose=0
    )

    # Train the model briefly for evaluation purposes
    print("Training the model...")
    model.learn(total_timesteps=50000)

    # Evaluate the model
    print("Evaluating the model...")
    eval_env = make_vec_env(lambda: TimeLimit(AdaptiveCruiseControlEnv(), max_episode_steps=1000), n_envs=1)
    obs = eval_env.reset()
    total_reward = 0.0
    done = [False]

    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward[0]

    eval_env.close()
    print(f"Trial completed. Total reward: {total_reward}")
    return total_reward

# Optimize hyperparameters using Optuna
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Retrieve the best hyperparameters
print("Best hyperparameters:", study.best_trial.params)
best_params = study.best_trial.params

# Train final model with optimized hyperparameters
print("Training final model with optimized hyperparameters...")
env = make_vec_env(lambda: TimeLimit(AdaptiveCruiseControlEnv(), max_episode_steps=1000), n_envs=1)
model = SAC(
    "MlpPolicy", env,
    gamma=best_params['gamma'],
    tau=best_params['tau'],
    learning_rate=best_params['learning_rate'],
    batch_size=best_params['batch_size'],
    ent_coef=best_params['ent_coef'],
    verbose=1
)

model.learn(total_timesteps=500000)
model.save("sac_acc_model_full")
wandb.save("sac_acc_model_full.zip")
print("Final model training complete and saved.")

# Load the trained model
print("Loading the trained model...")
model = SAC.load("sac_acc_model_full")

# Evaluate the final model
def evaluate_model(model, env):
    """Evaluate a trained model with a render loop."""
    print("Starting model evaluation...")
    obs = env.reset()
    done = [False]
    total_reward = 0.0

    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        env.render()
        time.sleep(0.1)  # Slow down rendering for visualization

    print(f"Model evaluation complete. Total reward: {total_reward}")
    return total_reward

# Setup evaluation environment
print("Setting up evaluation environment...")
eval_env = make_vec_env(lambda: TimeLimit(AdaptiveCruiseControlEnv(render_mode="human"), max_episode_steps=1000), n_envs=1)
pygame.init()
window = pygame.display.set_mode((200, 100))
pygame.display.set_caption("Press any key to start evaluation")
print("Press any key to start evaluation")

waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type in [pygame.KEYDOWN, pygame.QUIT]:
            waiting = False

pygame.quit()

# Start evaluation
print("Starting evaluation...")
evaluation_reward = evaluate_model(model, eval_env)
print(f"Evaluation completed. Total reward: {evaluation_reward}")

# Cleanup
print("Cleaning up resources...")
eval_env.close()
print("Done.")
