import sys
import numpy as np
from keras.models import load_model

from rl_capstone.blob import BlobEnv


def main(model_path: str) -> None:
    env = BlobEnv()
    model = load_model(model_path)
    state = env.reset()
    env.render()
    done = False
    while not done:
        action_probs, _ = model(np.array(state).reshape(1, -1), training=False)
        action = int(np.argmax(action_probs.numpy()[0]))
        state, _, done = env.step(action)
        env.render()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo_ppo_blob.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
