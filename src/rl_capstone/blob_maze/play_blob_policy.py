"""
play_blob_policy.py  –  Run a trained PPO policy deterministically.

examples
    python play_blob_policy.py                              # default model, 1 episode
    python play_blob_policy.py --model my_policy.zip        # custom file
    python play_blob_policy.py --episodes 5 --images        # 5 runs, RGB render
"""

import argparse
from stable_baselines3 import PPO
from blob_env import BlobEnv


def main(model_path: str, episodes: int, use_images: bool):
    env = BlobEnv(return_images=use_images)
    model = PPO.load(model_path, env=env)

    for ep in range(episodes):
        obs, _ = env.reset()
        total = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)  # greedy
            action = int(action) if isinstance(action, (list, tuple, bytes)) or hasattr(action, "shape") else action
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            env.render()
        print(f"Episode {ep + 1}: reward {total}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run trained Blob policy")
    p.add_argument("--model", default="ppo_blob_policy.zip", help="model filename")
    p.add_argument("--episodes", type=int, default=1, help="number of episodes")
    p.add_argument("--images", action="store_true", help="render RGB window instead of text grid")
    args = p.parse_args()
    main(args.model, args.episodes, args.images)
