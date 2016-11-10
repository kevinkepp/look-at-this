import random

from keras.optimizers import SGD

from lat.SimpleReward import RewardAtTheEnd, LinearReward
from lat.DeepQAgent import DeepQAgent
from lat.Evaluator import Evaluator
from lat.QAgent import QAgent
from lat.RandomAgent import RandomAgent
from lat.OldSimulator import Simulator, Actions
# from lat.Simulator import SimpleMatrixSimulator as Simulator, Actions
from lat.KerasMlpModel import KerasMlpModel

EPOCHS = 2000  # 3000
GRID_SIZE = 7
MAX_STEPS = pow(GRID_SIZE, 1.7)
BOUNDED = False

ACTIONS = Actions.all()
REWARD_LIN = LinearReward()
REWARD_AT_END = RewardAtTheEnd()

envs = []
names = []

# Random Agent (baseline)
agent = RandomAgent(ACTIONS)
# env = Simulator(agent, REWARD_LIN, GRID_SIZE, MAX_STEPS)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("Random")

# Q-Agent
ALPHA = 1  # optimal for deterministic env
GAMMA = 0.975  #
Q_INIT = lambda state, action: random.random()
agent = QAgent(ACTIONS, ALPHA, GAMMA, Q_INIT)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("QAgent")

# Deep Q-Agents
MODEL_IN_LAYER_SIZE = GRID_SIZE * GRID_SIZE
MODEL_OUT_LAYER_SIZE = len(ACTIONS)
EPSILON = 1
EPSILON_UPDATE = lambda e: e - 1 / EPOCHS if e > 0.05 else e

MODEL_HID_LAYER_SIZES = []
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgent(ACTIONS, GAMMA, EPSILON, EPSILON_UPDATE, model)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[]")

MODEL_HID_LAYER_SIZES = [50]
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgent(ACTIONS, GAMMA, EPSILON, EPSILON_UPDATE, model)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[50]")

MODEL_HID_LAYER_SIZES = [150, 75]
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgent(ACTIONS, GAMMA, EPSILON, EPSILON_UPDATE, model)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[150, 75]")

envs = [envs[3]]
names = [names[3]]
ev = Evaluator(envs, names, EPOCHS, grid="{0}x{1}".format(GRID_SIZE, GRID_SIZE), gamma=GAMMA)
ev.run(False)

if False:
	import matplotlib.pyplot as plt

	f = plt.figure()
	# hacky access to weights
	weights = envs[1]._agent._model._model.get_weights()[0]
	# print("Weights for " + names[1] + ":\n" + str(weights))
	for i in range(weights.shape[1]):
		ax = f.add_subplot(2, 2, i + 1)
		w = weights[:, i].reshape((GRID_SIZE, GRID_SIZE))
		ax.imshow(w, interpolation='nearest', aspect='auto', cmap="Blues")
		ax.set_title("Action " + str([a for a in Actions][i].name))
	plt.show()
