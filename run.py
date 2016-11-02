from random import random

import matplotlib.pyplot as plt
import numpy as np

from lat.QAgent import QLearningAgent
from lat.Simulator import Simulator

ALPHA = 1  # optimal for deterministic env
GAMMA = 0.9  # high for long history, but with >= 1 action values might diverge
GRID_SIZE = 20
MAX_STEPS = pow(GRID_SIZE, 1.5)

Q_INIT = lambda state, action: random()
EPOCHS = 1000

agent = QLearningAgent(ALPHA, GAMMA, Q_INIT)
env = Simulator(agent, MAX_STEPS, GRID_SIZE)

# train
print("Training " + str(EPOCHS) + " epochs")
succ = []
for i in range(EPOCHS):
	res = env.run()
	succ.append(1 if res else 0)
env.run()

# evaluate
print("Successful epochs out of last 10: " + str(sum(succ[-10:])))
batch_size = 5
batch_count = int(EPOCHS / batch_size)
batches = []
for i in range(0, batch_count):
	percent = sum(succ[i:i + batch_count]) / batch_count * 100.
	batches.append(percent)
plt.plot(np.arange(batch_count) * batch_size, batches)
plt.title("Training {0} epochs (avg over {1} epochs, alpha: {2}, gamma: {3})"
			.format(str(EPOCHS), str(batch_size), str(ALPHA), str(GAMMA)))
plt.xlabel("epochs")
plt.ylabel("% correct")
plt.ylim((0, 110))
plt.grid(True)
plt.show()
