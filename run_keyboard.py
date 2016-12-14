from random import random

from sft.agent.KeyboardQAgent import KeyboardQAgent
from sft.sim.OldSimulator import Simulator, Actions

ALPHA = 0.1
GAMMA = 0.1
MAX_STEPS = 25
GRID_SIZE = 5
Q_INIT = lambda state, action: random()

agent = KeyboardQAgent(Actions.all(), ALPHA, GAMMA, Q_INIT)
env = Simulator(agent, MAX_STEPS, GRID_SIZE)

res = env.run()
print("Successful: " + "Yes" if res else "No")
