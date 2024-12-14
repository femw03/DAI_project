import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import wandb  # Import wandb


class AdaptiveCruiseControlEnv(gym.Env):
    def __init__(self, config={}, render_mode=None):
        super(AdaptiveCruiseControlEnv, self).__init__()
        
        # Define the action space and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), 
                                            high=np.array([130, 130, 1, 110]), 
                                            dtype=np.float32)

        # Simulation parameters
        self.dt = 0.1
        self.max_speed = 30.0
        self.min_distance = 10.0
        self.max_distance = 100.0

        # Initialize pygame for visualization
        self.screen_width = 800
        self.screen_height = 200
        self.car_width = 50
        self.car_height = 20
        self.pixels_per_meter = 8  # Scale for visualization (8 pixels per meter)

        self.render_initialized = False
        self.clock = None
        self.screen = None
        self.episode = 0
        self.episode_reward = 0
        self.car_in_front = 1

        # Initialize wandb
        wandb.init(project="adaptive_cruise_control", config=config)
        
        # Reset environment variables
        self.reset()

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.agent_speed = np.random.uniform(0, 5)
        self.target_speed = 10.0
        self.current_target_speed = self.target_speed
        self.relative_distance = np.random.uniform(20, 50)
        self.last_action = 0.5  # Initialize to neutral action
        self.target_speed_update_interval = 250
        self.time_step_counter = 0
        wandb.log({"episode_reward": self.episode_reward})
        self.episode_reward = 0
        self.car_disappear_counter = 0  # Counter to manage car disappearance
        self.car_in_front = 1
        return np.array([self.agent_speed, self.max_speed, self.car_in_front, self.relative_distance], dtype=np.float32), {}

    def step(self, action):
        terminated = False
        self.last_action = action[0]  # Store the last action for rendering
        
        if action[0] <= 0.5:
            brake_intensity = (0.5 - action[0]) * 2
            self.agent_speed = max(0.0, self.agent_speed - brake_intensity * 10.0 * self.dt)
        else:
            throttle_intensity = (action[0] - 0.5) * 2
            self.agent_speed = self.agent_speed + throttle_intensity * 2.0 * self.dt

        if self.agent_speed > self.max_speed + 10:
            terminated = True

        # Update target speed gradually
        if self.time_step_counter % self.target_speed_update_interval == 0:
            self.target_speed = np.random.uniform(10, 30)
            self.target_speed_update_interval = 250

        # Gradually adjust current target speed towards target speed
        speed_diff = self.target_speed - self.current_target_speed
        speed_adjustment = np.sign(speed_diff) * min(abs(speed_diff), 0.1)  # Gradual change
        self.current_target_speed += speed_adjustment

        self.time_step_counter += 1

        # Introduce car disappearance randomly
        if self.time_step_counter % 600 == 0:  # Every 1000 steps, the car might disappear
            if np.random.rand() < 0.5:  # 50% chance to disappear
                self.car_in_front = 0
                self.current_target_speed = 0.0
                self.car_disappear_counter = 500  # Car disappears for 500 steps

        if self.car_disappear_counter > 0:
            self.car_disappear_counter -= 1
            if self.car_disappear_counter == 0:
                self.current_target_speed = self.target_speed  # Car reappears
                self.car_in_front = 1

        self.relative_distance += (self.current_target_speed - self.agent_speed) * self.dt

        reward = self._get_reward()

        self.episode_reward += reward

        truncated = self.relative_distance > self.max_distance
        terminated = self.relative_distance < 0.5

        # Log data to wandb
        wandb.log({
            "episode": self.episode,
            "distance": self.relative_distance,
            "agent_speed": self.agent_speed,
            "target_speed": self.current_target_speed,
            "reward": reward,
            "action": action[0]
        })
        observation = np.array([self.agent_speed, self.max_speed, self.car_in_front, self.relative_distance], dtype=np.float32)
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return
        
        # Initialize pygame if not done yet
        if not self.render_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Adaptive Cruise Control Simulation")
            self.clock = pygame.time.Clock()
            self.render_initialized = True

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Calculate positions for visualization
        target_x = 50
        agent_x = target_x + self.relative_distance * self.pixels_per_meter

        # Draw target car (only if it hasn't disappeared)
        if self.current_target_speed > 0:
            pygame.draw.rect(self.screen, (0, 0, 255), (target_x, self.screen_height // 2, self.car_width, self.car_height))

        # Draw agent car
        pygame.draw.rect(self.screen, (255, 0, 0), (agent_x, self.screen_height // 2, self.car_width, self.car_height))

        # Determine action type based on last action value
        action_type = "Throttle" if self.last_action > 0.5 else "Brake"

        # Draw speed, distance, and action text
        font = pygame.font.Font(None, 36)
        distance_text = font.render(f"Distance: {self.relative_distance:.1f} m", True, (0, 0, 0))
        agent_speed_text = font.render(f"Agent Speed: {self.agent_speed:.1f} m/s", True, (0, 0, 0))
        action_text = font.render(f"Action: {action_type} ({self.last_action:.2f})", True, (0, 0, 0))  # Display actual action value
        target_speed_text = font.render(f"Lead vehicle speed: {self.current_target_speed}", True, (0, 0, 0))

        self.screen.blit(distance_text, (10, 10))
        self.screen.blit(agent_speed_text, (10, 50))
        self.screen.blit(action_text, (10, 90))
        self.screen.blit(target_speed_text, (10, 130))

        # Update display
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_initialized:
            pygame.quit()
            self.render_initialized = False

    def _get_reward(self) -> float:
        """
        Calculate reward based on the action and the current environment state.
        Reward based on speed and distance, with constraints on safe driving.
        """
        speed_limit = self.max_speed
        current_speed = self.agent_speed
        speed_margin = speed_limit*0.15
 
        # Constants
        safe_distance_margin = 0.25
        max_safe_distance = 100
 
        # Default reward
        reward = 0

        # Speed Reward Calculation 
        speed_reward = 0 
        max_speed = speed_limit + speed_margin 
        if self.current_target_speed == 0: # No car in front 
            if current_speed == 0: 
                speed_reward = 0 # No reward for being stationary 
            elif 0 < current_speed <= speed_limit: 
                speed_reward = min(1, current_speed / speed_limit) # Linearly scale up to the speed limit 
            elif speed_limit < current_speed < max_speed: 
                speed_reward = max(0, (max_speed - current_speed) / (max_speed - speed_limit)) # Linearly ramp down 
            elif current_speed >= max_speed: 
                speed_reward = 0 # No reward if speed exceeds or equals max_speed

        safe_distance_reward = 0
        if current_speed > 1:
            safe_distance = 2 * (
                current_speed / 3.6
            )  # Safe distance = 2 seconds of travel in m/s
        else:
            safe_distance = 1

        lower_bound = safe_distance - safe_distance_margin
        upper_bound = safe_distance + safe_distance_margin

        if lower_bound <= self.relative_distance <= upper_bound:
            safe_distance_reward = 1  # Perfect safe distance
        elif self.relative_distance < lower_bound:
            safe_distance_reward = max(
                0, self.relative_distance / lower_bound
            )  # Linearly decrease to 0
        elif self.relative_distance > upper_bound:
            safe_distance_reward = max(
                0,
                1
                - (self.relative_distance - upper_bound)
                / (max_safe_distance - upper_bound),
            )
 
        reward = speed_reward if self.current_target_speed == 0 else safe_distance_reward 
        # Ensure reward is in range [0, 1] 
        reward = np.clip(reward, 0, 1)
 
        # Logging for debugging and analysis
        wandb.log(
            {"speed_reward": speed_reward, 
             "safe_distance_reward": safe_distance_reward, 
             "reward": reward, 
             "speed_limit": speed_limit, 
             "current_speed": current_speed 
             }
        )
        return reward