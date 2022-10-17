# RL_Capstone
Capstone Project for GA Using Reinforcement Learning



### Description

This goal of this project was to program a basic version of tetris and implement a basic version of reinforcement learning so that the agent could play the game and clearl lines (preferably at a high level).

### Scenario

DeepMind has done some incredible work in the area of reinforcement learning in the past decade, mastering games like Chess, Go and a variety of Atari games. Beyond games, they recently solved protein folding with AlphaFold demonstrating that learning optimal policies to solve complex problems is not just limited to games. I wanted to learn some of the fundamentals of RL by implementing an agent to play the game Tetris, simple to understand but complicated enough to be a great final project.

#### Executive Summary

This project consists of two main steps: coding and updating the actual game tetris and implementing the RL agent to try to learn optimal poicies. 

The problem statement for this project is: _Can a reinforcment agent learn to play Tetris, preferably optimally?_

The main data that was acquired was sequences of potential actions and corresponding properties of the game state such as total height, bumpiness,  etc. The task of the agent was to make predictions on how well it evaluated each position in terms of future reward (clearing lines and scoring points).


#### Game Programming: 
Having already programmed Tetris in MATLAB, I started with a very vanilla version of the game programed from https://levelup.gitconnected.com/writing-tetris-in-python-2a16bddb5318, mainly to learn and understand how timers and rendering was done in python. Several important features were added including debugging buttons to allow user to more easily recreate situations and check functionality, SRS rotation system, shadows for blocks to see where they land, and the ability to switch from human player to computer. Currently, the computer only makes random moves, but the goal is to have the computer play optimal moves chosen by the agent (hopefully to potentially rescue the user if they are in a pinch).


#### RL:

After working with OpenAI gym and stable baselines, I decided I'd prefer to use a custom agent to play the game with self defined neural networks as it was more clear what exactly was going on behind the scenes and seemed better to debug. I followed a similar approach to https://github.com/nuno-faria/tetris-ai/blob/master/README.md and adopted their strategy of simplifying the game state and using a DQN network. Although, my states and observations were defined differently and had many issues trying to get a similar network to train and run. After several hours of training, the notebook seemed to crash and after retraining- the agent appeared to learn a few things but still seemed to need to do a significant amount of exploring to learn the true q-values of different state action pairs.


#### Conclusions
Most of my time was spent solving logic and coding issues with this project. I was not able to start modeling until significantly later than desired and I believe with more training time the agent's performance will dramatically improve. I also hope to implement several features to make the game more enjoyable such as sound and the ability to form gold/silver blocks. I plan to continue researching and learning about machine learning and specifically reinforcement learning to develop better methods for helping the agent learn more distant rewards such as creating four line clears (Tetrises) and hopefully will get the agent to learn how to create gold/silver blocks.