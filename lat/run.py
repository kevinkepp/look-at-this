from random import random

from lat.QLearningAgent import QLearningAgent
from lat.Simulator import Simulator

ALPHA = 0.1
GAMMA = 0.1
MAX_STEPS = 25
Q_INIT = lambda state, action: random()

agent = QLearningAgent(ALPHA, GAMMA, Q_INIT)
env = Simulator(agent, MAX_STEPS)

# train
EPOCHS = 100
env.train(EPOCHS)
