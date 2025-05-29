import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse

# --- Training the agent with Stable Baselines3 PPO ---
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# --- Custom Blob Environment, now Gymnasium-compatible ---
class BlobEnv(gym.Env):
    """Custom 10x10 Grid environment with a player, food, and enemy."""

    # Environment constants
    SIZE = 10
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300  # negative reward for hitting enemy
    FOOD_REWARD = 25  # positive reward for reaching food
    MAX_STEPS = 200

    def __init__(self, return_images: bool = False):
        super().__init__()
        self.return_images = return_images
        # Define action and observation spaces
        # 9 discrete actions: 8 directions (N, S, E, W and diagonals) + 1 "stay" (for example).
        self.action_space = spaces.Discrete(9)
        if self.return_images:
            # Observation is an image: 10x10 RGB grid
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.SIZE, self.SIZE, 3), dtype=np.uint8)
        else:
            # Observation is a 4-dimensional vector: (dx_to_food, dy_to_food, dx_to_enemy, dy_to_enemy)
            # These deltas range from -9 to 9. Define space accordingly.
            low = np.array([-self.SIZE] * 4, dtype=np.int32)  # e.g. -10 in each dimension
            high = np.array([self.SIZE] * 4, dtype=np.int32)  # e.g. +10 in each dimension
            # Use dtype float32 for compatibility with PPO policies
            self.observation_space = spaces.Box(
                low=low.astype(np.float32), high=high.astype(np.float32), shape=(4,), dtype=np.float32
            )
        # Initialize state variables
        self.player_pos = None
        self.food_pos = None
        self.enemy_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        # Randomly place player, food, and enemy such that they don't overlap
        # Using (x, y) coordinates in [0, SIZE-1]
        # (Gymnasium reset can take a seed and options, we ignore or use seed for RNG)
        super().reset(seed=seed)
        self.steps = 0
        # Random positions
        self.player_pos = np.random.randint(0, self.SIZE, size=2)
        self.food_pos = np.random.randint(0, self.SIZE, size=2)
        # Ensure food not at player position
        while np.array_equal(self.food_pos, self.player_pos):
            self.food_pos = np.random.randint(0, self.SIZE, size=2)
        self.enemy_pos = np.random.randint(0, self.SIZE, size=2)
        # Ensure enemy not at player or same as food
        while np.array_equal(self.enemy_pos, self.player_pos) or np.array_equal(self.enemy_pos, self.food_pos):
            self.enemy_pos = np.random.randint(0, self.SIZE, size=2)
        # Construct initial observation
        obs = self._get_obs()
        # Gymnasium reset returns (obs, info)
        return obs, {}  # no extra info

    def step(self, action):
        # Apply action: define movement vectors for actions 0-8.
        # We map action index to a movement (dx, dy). For example:
        moves = {
            0: np.array([0, 0]),  # stay
            1: np.array([0, 1]),  # up
            2: np.array([0, -1]),  # down
            3: np.array([1, 0]),  # right
            4: np.array([-1, 0]),  # left
            5: np.array([1, 1]),  # up-right (diagonal)
            6: np.array([1, -1]),  # down-right
            7: np.array([-1, 1]),  # up-left
            8: np.array([-1, -1]),  # down-left
        }
        # Update player position with the chosen move
        move = moves.get(action, np.array([0, 0]))
        self.player_pos = np.clip(self.player_pos + move, 0, self.SIZE - 1)  # stay within bounds
        self.steps += 1

        # Compute reward
        reward = 0
        # Check if player hits enemy
        if np.array_equal(self.player_pos, self.enemy_pos):
            reward = -self.ENEMY_PENALTY
        # Check if player reaches food
        elif np.array_equal(self.player_pos, self.food_pos):
            reward = +self.FOOD_REWARD
        else:
            # Small negative reward for each move to encourage efficiency
            reward = -self.MOVE_PENALTY

        # Check termination conditions
        terminated = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY:
            terminated = True  # episode successful or failed
        truncated = False
        if self.steps >= self.MAX_STEPS:
            # Time limit reached without success/failure
            truncated = True

        # Get next observation
        obs = self._get_obs()
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Helper to construct observation based on current positions."""
        if self.return_images:
            # Build a 10x10 RGB image representation
            env_grid = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
            # Color coding: e.g., blue for player, green for food, red for enemy
            player_color = (0, 0, 255)  # (BGR in cv2, but as RGB here)
            food_color = (0, 255, 0)
            enemy_color = (255, 0, 0)
            px, py = self.player_pos
            fx, fy = self.food_pos
            ex, ey = self.enemy_pos
            env_grid[px, py] = player_color
            env_grid[fx, fy] = food_color
            env_grid[ex, ey] = enemy_color
            return env_grid  # as an array (uint8 image)
        else:
            # Return vector of differences (player - food and player - enemy)
            dx_food = self.player_pos[0] - self.food_pos[0]
            dy_food = self.player_pos[1] - self.food_pos[1]
            dx_enemy = self.player_pos[0] - self.enemy_pos[0]
            dy_enemy = self.player_pos[1] - self.enemy_pos[1]
            # Convert to float32 for consistency
            return np.array([dx_food, dy_food, dx_enemy, dy_enemy], dtype=np.float32)

    def render(self):
        """Optional: Render the grid for visualization (prints text or shows image)."""
        if self.return_images:
            # Enlarge the image and display (this requires cv2 or PIL to actually show a window)
            from PIL import Image

            img = Image.fromarray(self._get_obs(), "RGB")
            img = img.resize((300, 300), Image.NeAREST)  # enlarge for visibility
            img.show()  # This will open the image; on headless environment, you might use plt.imshow.
        else:
            # Simple text render: print positions on a grid
            grid = [["." for _ in range(self.SIZE)] for __ in range(self.SIZE)]
            px, py = self.player_pos
            grid[px][py] = "P"
            fx, fy = self.food_pos
            grid[fx][fy] = "F"
            ex, ey = self.enemy_pos
            grid[ex][ey] = "E"
            print("\n".join(" ".join(row) for row in grid))
            print(f"Step: {self.steps}, Player: {self.player_pos}, Food: {self.food_pos}, Enemy: {self.enemy_pos}")


def train(total_timesteps: int, save_as: str = "ppo_blob_policy.zip") -> None:
    """Train PPO on BlobEnv."""
    env = BlobEnv(return_images=False)
    check_env(env, warn=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./blob_tensorboard/",
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(save_as)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on the Blob grid task")
    parser.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500 000)",
    )
    parser.add_argument(
        "--model-out",
        default="ppo_blob_policy.zip",
        help="Filename for the saved policy",
    )
    args = parser.parse_args()

    train(total_timesteps=args.steps, save_as=args.model_out)
