"""
train_blob_sb3.py  –  Train PPO on BlobEnv.

examples
    python train_blob_sb3.py                         # 500 k steps, no render
    python train_blob_sb3.py --steps 200000          # custom steps
    python train_blob_sb3.py --watch 10000           # show one episode every 10 k steps
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from blob_env import BlobEnv


def run_one_episode(model, env, deterministic=True):
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()


def train(total_timesteps: int, save_as: str, watch_every: int | None):
    env = BlobEnv(return_images=False)
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./blob_tensorboard/",
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
    )

    if watch_every is None:
        model.learn(total_timesteps=total_timesteps)
    else:
        # train in chunks and render after each chunk
        for _ in range(0, total_timesteps, watch_every):
            model.learn(total_timesteps=watch_every, reset_num_timesteps=False)
            run_one_episode(model, env)  # visual check

    model.save(save_as)
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PPO on Blob grid")
    p.add_argument("--steps", type=int, default=500_000, help="total timesteps (default 500k)")
    p.add_argument("--model-out", default="ppo_blob_policy.zip", help="output filename")
    p.add_argument("--watch", type=int, help="render one episode every N timesteps")
    args = p.parse_args()
    train(args.steps, args.model_out, args.watch)
