import argparse
import os
import time
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from rl_capstone.blob_maze.OLD_blob import BlobEnv


def discount_rewards(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted returns."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


class PPOAgent:
    """Minimal PPO agent for the Blob environment."""

    def __init__(
        self,
        state_size: int = 4,
        action_size: int = BlobEnv.ACTION_SPACE_SIZE,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        lr: float = 0.001,
        update_epochs: int = 5,
    ) -> None:
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        inputs = keras.Input(shape=(state_size,))
        hidden = layers.Dense(128, activation="relu")(inputs)
        action = layers.Dense(action_size, activation="softmax")(hidden)
        value = layers.Dense(1)(hidden)
        self.model = keras.Model(inputs=inputs, outputs=[action, value])
        self.optimizer = keras.optimizers.Adam(lr)
        self.action_size = action_size

    def act(self, state: List[float]):
        state = np.array(state)[None, :]
        action_probs, _ = self.model(state, training=False)
        action_probs = np.squeeze(action_probs.numpy())
        action = np.random.choice(self.action_size, p=action_probs)
        log_prob = np.log(action_probs[action] + 1e-8)
        return int(action), float(log_prob)

    def train_episode(
        self,
        states: List[List[float]],
        actions: List[int],
        log_probs: List[float],
        rewards: List[float],
    ) -> None:
        returns = discount_rewards(rewards, self.gamma)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        old_log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # Advantage estimation
        _, values = self.model(states, training=False)
        values = tf.squeeze(values)
        advantages = returns - values

        action_onehot = tf.one_hot(actions, self.action_size)
        dataset = tf.data.Dataset.from_tensor_slices((states, action_onehot, old_log_probs, returns, advantages)).batch(
            len(states)
        )

        for _ in range(self.update_epochs):
            for batch_states, batch_actions, batch_old_log, batch_returns, batch_adv in dataset:
                with tf.GradientTape() as tape:
                    probs, values = self.model(batch_states, training=True)
                    values = tf.squeeze(values)
                    new_log = tf.math.log(tf.reduce_sum(probs * batch_actions, axis=1) + 1e-8)
                    ratio = tf.exp(new_log - batch_old_log)
                    clipped = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * batch_adv, clipped * batch_adv))
                    value_loss = tf.reduce_mean((batch_returns - values) ** 2)
                    loss = policy_loss + 0.5 * value_loss
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def train(episodes: int = 500, render: bool = False, model_dir: str = "ppo_blob_models") -> None:
    env = BlobEnv()
    agent = PPOAgent()
    os.makedirs(model_dir, exist_ok=True)

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        states, actions, log_probs, rewards = [], [], [], []
        ep_reward = 0.0
        if render:
            env.render()
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            ep_reward += reward
            state = next_state
            if render:
                env.render()
        agent.train_episode(states, actions, log_probs, rewards)
        print(f"Episode {episode}: reward={ep_reward:.2f}")
        if episode % 50 == 0:
            filename = f"ppo_blob_{episode}_{int(time.time())}.keras"
            agent.model.save(os.path.join(model_dir, filename))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent in the Blob environment")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    args = parser.parse_args()
    train(episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
