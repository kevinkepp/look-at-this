from random import random

from lat.KeyboardQAgent import KeyboardQAgent
from lat.QAgent import QLearningAgent
from lat.Simulator import Simulator

ALPHA = 0.1
GAMMA = 0.1
MAX_STEPS = 25
GRID_SIZE = 5
Q_INIT = lambda state, action: random()

agent = KeyboardQAgent(ALPHA, GAMMA, Q_INIT)
env = Simulator(agent, MAX_STEPS, GRID_SIZE)

res = env.run()
print("Successful: " + "Yes" if res else "No")
