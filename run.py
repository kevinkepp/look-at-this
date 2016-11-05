import random

import matplotlib.pyplot as plt
import numpy as np

from lat.DeepQAgent import DeepQAgent
from lat.QAgent import QAgent
from lat.Simulator import Simulator, Actions
from lat.KerasMlpModel import KerasMlpModel

GRID_SIZE = 3
MAX_STEPS = pow(GRID_SIZE, 2)
EPOCHS = 250

ALPHA = 1  # optimal for deterministic env
GAMMA = 0.9  # high for long history, but with >= 1 action values might diverge
Q_INIT = lambda state, action: random.random()
q_agent = QAgent(Actions, ALPHA, GAMMA, Q_INIT)

EPSILON = 1
EPSILON_UPDATE = lambda e: e - 1 / EPOCHS if e > 0.1 else e
ACTIONS_COUNT = len(Actions.all())
MODEL_IN_LAYER_SIZE = GRID_SIZE * GRID_SIZE
MODEL_HID_LAYER_SIZE = 10
MODEL_OUT_LAYER_SIZE = ACTIONS_COUNT
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZE, MODEL_OUT_LAYER_SIZE)
deep_q_agent = DeepQAgent(GAMMA, EPSILON, EPSILON_UPDATE, ACTIONS_COUNT, model)

env = Simulator(deep_q_agent, MAX_STEPS, GRID_SIZE)

# train
print("Training " + str(EPOCHS) + " epochs")
res = env.run(EPOCHS)
succ = [1 if r else 0 for r in res]
print("Res: " + str(succ))

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
plt.ylim((0, 101))
plt.grid(True)
plt.show()
