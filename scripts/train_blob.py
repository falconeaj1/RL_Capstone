import argparse
import os
import random
import time
from collections import deque

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm

from rl_capstone.blob_maze.OLD_blob import BlobEnv


class ModifiedTensorBoard(TensorBoard):
    """TensorBoard callback that keeps a single writer open."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        """Initialize training and validation counters without assigning the model."""
        super().set_model(model)
        self._train_dir = os.path.join(self._log_write_dir, "train")
        # Newer versions of Keras expose counters on the model; fall back to 0 if absent
        self._train_step = getattr(self.model, "_train_counter", tf.Variable(0, dtype="int64"))
        self._val_dir = os.path.join(self._log_write_dir, "validation")
        self._val_step = getattr(self.model, "_test_counter", tf.Variable(0, dtype="int64"))
        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**(logs or {}))

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 32
UPDATE_TARGET_EVERY = 2
MODEL_NAME = "simple"


class DQNAgent:
    STATE_SIZE = 4
    MODEL_INPUT_SIZE = (-1, 1, STATE_SIZE)

    def __init__(self) -> None:
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def _create_model(self) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(1, self.STATE_SIZE)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(BlobEnv.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition) -> None:
        self.replay_memory.append(transition)

    def get_qs(self, state) -> np.ndarray:
        s = np.array(state).reshape(self.MODEL_INPUT_SIZE)
        return self.model.predict(np.array(s), verbose=0)[0]

    def train(self) -> None:
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([t[0] for t in minibatch]).reshape(self.MODEL_INPUT_SIZE)
        new_states = np.array([t[3] for t in minibatch]).reshape(self.MODEL_INPUT_SIZE)
        current_qs_list = self.model.predict(current_states, verbose=0)
        future_qs_list = self.target_model.predict(new_states, verbose=0)
        X = []
        y = []
        for index, (state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[0, action] = new_q
            X.append(state)
            y.append(current_qs)
        self.target_update_counter += 1
        self.model.fit(
            np.array(X).reshape(self.MODEL_INPUT_SIZE),
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if self.target_update_counter % UPDATE_TARGET_EVERY == 0 else None,
        )
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def train(episodes: int = 200, show_preview: bool = False) -> None:
    env = BlobEnv()
    agent = DQNAgent()
    epsilon = 1.0
    percent_to_min = 0.95
    min_epsilon = 0.1
    epsilon_decay = np.exp(np.log(min_epsilon) / (episodes * percent_to_min))
    aggregate_stats_every = 1
    best_avg_reward = -201
    ep_rewards = []
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    if not os.path.isdir("models"):
        os.makedirs("models")
    for episode in tqdm(range(1, episodes + 1), ascii=True, unit="episodes"):
        agent.tensorboard.step = episode
        episode_reward = 0
        current_state = env.reset()
        done = False
        if show_preview and not episode % aggregate_stats_every:
            env.render()
        while not done:
            if np.random.random() > epsilon:
                action = int(np.argmax(agent.get_qs(current_state)))
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            new_state, reward, done = env.step(action)
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state
            if show_preview and not episode % aggregate_stats_every:
                env.render()
        agent.train()
        ep_rewards.append(episode_reward)
        if episode % aggregate_stats_every == 0 or episode == 1:
            avg_reward = sum(ep_rewards[-aggregate_stats_every:]) / len(ep_rewards[-aggregate_stats_every:])
            min_reward = min(ep_rewards[-aggregate_stats_every:])
            max_reward = max(ep_rewards[-aggregate_stats_every:])
            agent.tensorboard.update_stats(
                reward_avg=avg_reward,
                reward_min=min_reward,
                reward_max=max_reward,
                epsilon=epsilon,
            )
            if avg_reward >= best_avg_reward:
                best_avg_reward = avg_reward
                agent.model.save(
                    f"models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras"
                )
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            epsilon = max(min_epsilon, epsilon)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent in the Blob environment")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    args = parser.parse_args()
    train(episodes=args.episodes, show_preview=args.render)


if __name__ == "__main__":
    main()