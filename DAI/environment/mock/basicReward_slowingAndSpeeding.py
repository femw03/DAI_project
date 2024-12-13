import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import wandb  # Import wandb

class AdaptiveCruiseControlEnv(gym.Env):
    def __init__(self, config={}, render_mode=None):
        super(AdaptiveCruiseControlEnv, self).__init__()
        
        # Define the action space and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, -100.0]), 
                                            high=np.array([45.0, 30.0, 1.0, 0.0]), 
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
        return np.array([self.agent_speed, self.max_speed, 1, self.relative_distance], dtype=np.float32), {}

    def step(self, action):
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

        self.relative_distance += (self.current_target_speed - self.agent_speed) * self.dt

        if self.relative_distance <= self.min_distance:
            reward = -2
            terminated = True
        elif self.relative_distance < 15.0:
            reward = 1
            terminated = False
        else:
            reward = -0.1 * (self.relative_distance / self.max_distance)
            terminated = False

        self.episode_reward += reward

        truncated = self.relative_distance > self.max_distance

        # Log data to wandb
        wandb.log({
            "episode": self.episode,
            "distance": self.relative_distance,
            "agent_speed": self.agent_speed,
            "target_speed": self.current_target_speed,
            "reward": reward,
            "action": action[0]
        })
        observation = np.array([self.agent_speed, self.max_speed, 1, self.relative_distance], dtype=np.float32)
        
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

        # Draw target car
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