# RL_Capstone
Reinforcement Learning Project
- Simple Maze Game
- Tetris

<<<<<<< HEAD
Instructions are for running from main area of repo.
=======
### Setup

Install dependencies with [uv](https://github.com/astral-sh/uv). The project
provides a `uv.lock` file for reproducible installations. Running the following
command installs the locked packages and the project itself:
```
uv pip install --system -e .
```

Codex disables network access after the setup phase, so place the above command
in a `setup.sh` script to ensure packages are available when the environment goes
offline.

### Usage
>>>>>>> main

## Installation
If you don't have `uv` installed:
### For macOS or Linux
brew install uv
### OR via pipx (works everywhere)
pipx install uv

### Create UV env
uv venv .venv              # makes ./venv folder
source .venv/bin/activate  # activate it
uv pip install -r pyproject.toml  # install dependencies

## Blob Maze Usage

Original game can be run with:
`python src/rl_capstone/blob_maze/OLD_blob.py`
This just randomly moves a blob that will be trained to go for green food and avoid red enemy.

I spent a while to try to get blob to learn, but blob was stubborn. The following deep-q network has some bugs.
`python scripts/train_blob.py`

Using ChatGPT to help learn about proper set up, was able to get a refactored version of game with simpler graphics. Will look into matching former style.

To train a PPO agent on the simple Blob environment, execute:
`src/rl_capstone/blob_maze/train_blob_sp3.py`
Use `--help` to see optional flags for training.
You can skip training and see results by running a greedy policy with:
`src/rl_capstone/blob_maze/play_blob_policy.py`
Use `--help` to see optional flags for playback.

The PPO agent receives a small positive reward when moving closer to the food 
and a negative reward when moving away. Main trained model is aved at `ppo_blob_policy.zip`
TODO: blob model needs moved to better spot

## Tetris
To play Tetris, execute:
`python src/rl_capstone/play_tetris.py --render-mode human`

The training of models was wokred on in `tetris_RL.ipynb`, along with notes of progress along the way. Model has not successfully been implemented/trained and a more thorough investigation of training needs to be implemented..

TODO: move progress notes into a changelog file

### Description

This goal of this project was to program a basic version of tetris and implement a basic version of reinforcement learning so that the agent could play the game and clear lines (preferably at a high level). An easier game of a straightforward maze (agent moves towards green object and avoids red object), was also included to help figure out details of training and deep-Q networks.

### Scenario

DeepMind has done some incredible work in the area of reinforcement learning in the past decade, mastering games like Chess, Go and a variety of Atari games. Beyond games, they recently solved protein folding with AlphaFold demonstrating that learning optimal policies to solve complex problems is not just limited to games. This project is an attempt to learn some of the fundamentals of RL by implementing an agent to play the game Tetris, simple to understand but complicated enough to be a great project.

#### Executive Summary

This project consists of two main steps: coding and updating the actual game tetris and implementing the RL agent to try to learn optimal policies. 

The problem statement for this project is: _Can a reinforcement agent learn to play Tetris, preferably optimally?_

The main data that was acquired was sequences of potential actions and corresponding properties of the game state such as total height, bumpiness,  etc. The task of the agent was to make predictions on how well it evaluated each position in terms of future reward (clearing lines and scoring points).


#### Game Programming: 
Having already programmed Tetris in MATLAB, I started with a very vanilla version of the game programed from https://levelup.gitconnected.com/writing-tetris-in-python-2a16bddb5318, mainly to learn and understand how timers and rendering was done in python. Several important features were added including debugging buttons to allow user to more easily recreate situations and check functionality, SRS rotation system, shadows for blocks to see where they land, and the ability to switch from human player to computer where the computer only makes random moves, but the goal is to have the computer play optimal moves chosen by the agent (hopefully to potentially rescue the user if they are in a pinch).


#### RL:

I followed a similar approach to https://github.com/nuno-faria/tetris-ai/blob/master/README.md and adopted their strategy of simplifying the game state and using a DQN network. Although, my states and observations were defined differently and had many issues trying to get a similar network to train and run. After several hours of training, the notebook seemed to crash and after retraining- the agent appeared to learn a few things but still seemed to need to do a significant amount of exploring to learn the true q-values of different state action pairs.


#### Conclusions
Most of my time was spent solving logic and coding issues with this project. I was not able to start modeling until significantly later than desired and I believe with more training time the agent's performance will dramatically improve. I also hope to implement several features to make the game more enjoyable such as sound and the ability to form gold/silver blocks. I plan to continue researching and learning about machine learning and specifically reinforcement learning to develop better methods for helping the agent learn more distant rewards such as creating four line clears (Tetrises) and hopefully will get the agent to learn how to create gold/silver blocks.
