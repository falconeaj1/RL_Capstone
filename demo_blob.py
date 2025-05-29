import numpy as np
from keras.models import load_model
from rl_capstone.blob_maze.OLD_blob import BlobEnv

model = load_model("models/simple____24.00max___24.00avg___24.00min__1748412714.keras")  # replace with the actual filename
env = BlobEnv()
state = env.reset()
env.render()
done = False
while not done:
    action = int(np.argmax(model.predict(
        np.array(state).reshape(-1, 1, 4), verbose=0)))
    state, reward, done = env.step(action)
    env.render()