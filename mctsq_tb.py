"""
----------------------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

mctsq_tb: 
> Starts the tensorboard server for monitoring of RL training results
----------------------------------------------------------------------------------------------------
"""

import os

# ----------------------------------------------------------------------------------------------------------------------
print("Tensorboard URL...")
os.system('tensorboard --logdir=tensorboard/')
