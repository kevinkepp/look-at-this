import random
import numpy as np
from lat.DeepQAgentReplay import DeepQAgentReplay
from lat.SimpleReward import RewardAtTheEnd, LinearReward
from lat.DeepQAgent import DeepQAgent
from lat.Evaluator import Evaluator
from lat.QAgent import QAgent
from lat.RandomAgent import RandomAgent
from lat.OldSimulator import Simulator, Actions
# from lat.Simulator import SimpleMatrixSimulator as Simulator, Actions
from lat.KerasMlpModel import KerasMlpModel

## Global parameters
EPOCHS = 1000  # runs/games
GRID_SIZE = 7
MAX_STEPS = pow(GRID_SIZE, 1.7)  # max steps per run/game
BOUNDED = False  # false means terminate on out of bounds, true means no out of bounds possible

## Agent parameters
# learning rate for Q-Agents
ALPHA = 1  # 1 is optimal for deterministic env
# discount factor
GAMMA = 0.99  # (see doi:10.1038/nature14236)
# epsilon-greedy strategy: agent chooses random action if rand() < epsilon
EPSILON_START = 1
# minimum epsilon in order to always guarantee a random influence (see doi:10.1038/nature14236)
EPSILON_MIN = 0.1
# linear decrease over number of epochs but minimum min_e
EPSILON_UPDATE_LIN = lambda n: max(1 - n / EPOCHS, EPSILON_MIN)
# linear decrease to n_min
_EPSILON_UPDATE_LIN_FAST = lambda n, n_min: max(1 - n / n_min, EPSILON_MIN)
# linear decrease to epochs/50, works for large EPOCHS (see doi:10.1038/nature14236)
EPSILON_UPDATE_ATARI = lambda n: _EPSILON_UPDATE_LIN_FAST(n, EPOCHS / 50)
# linear decrease to epochs/20, like atari but for smaller EPOCHS
EPSILON_UPDATE_ATARI_SMALL = lambda n: _EPSILON_UPDATE_LIN_FAST(n, EPOCHS / 20)
# multiply by fixed factor but minimum min_e
_EPSILON_UPDATE_FACTOR = lambda n, f: max(np.power(f, n), EPSILON_MIN)
# custom update function with convex shape, adapted -log(x) with f(1) = 1 and f(EPOCHS) = 0
def EPSILON_UPDATE_KEV(n):
	a = (np.exp(-1) * EPOCHS - 1) / (1 - np.exp(-1))
	return max(-np.log((n + a) / (EPOCHS + a)), EPSILON_MIN)
# q value init (QAgent)
Q_INIT = lambda state, action: random.random()
# replay batch size (DeepQAgentReplay)
REPLAY_BATCH_SIZE_ATARI = 32
# replay buffer (DeepQAgentReplay)
# buffer size EPOCHS/50  (see doi:10.1038/nature14236)
REPLAY_BUFFER_ATARI = EPOCHS / 50
# like atari but for smaller EPOCHS
REPLAY_BUFFER_ATARI_SMALL = EPOCHS / 20

## Environment parameters
ACTIONS = Actions.all()
# different reward functions
REWARD_LIN = LinearReward()
REWARD_AT_END = RewardAtTheEnd()

## Setup agents and environments
envs = []
names = []
# Random Agent (baseline)
agent = RandomAgent(ACTIONS)
# env = Simulator(agent, REWARD_LIN, GRID_SIZE, MAX_STEPS)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("Random")
# Q-Agent
agent = QAgent(ACTIONS, ALPHA, GAMMA, Q_INIT)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("QAgent")
# Deep Q-Agents
MODEL_IN_LAYER_SIZE = GRID_SIZE * GRID_SIZE
MODEL_OUT_LAYER_SIZE = len(ACTIONS)
MODEL_HID_LAYER_SIZES = []
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgent(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[]")

model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model,
						 REPLAY_BATCH_SIZE_ATARI, REPLAY_BUFFER_ATARI_SMALL)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[] Replay")

MODEL_HID_LAYER_SIZES = [50]
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgent(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[50]")

model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model,
						 REPLAY_BATCH_SIZE_ATARI, REPLAY_BUFFER_ATARI_SMALL)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[50] Replay")

MODEL_HID_LAYER_SIZES = [150, 75]
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgent(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model)
env = Simulator(agent, REWARD_LIN, GRID_SIZE, GRID_SIZE, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[150, 75]")

include = [2, 3, 4, 5]
envs = [envs[i] for i in include]
names = [names[i] for i in include]
ev = Evaluator(envs, names, EPOCHS, grid="{0}x{1}".format(GRID_SIZE, GRID_SIZE), gamma=GAMMA)
ev.run(True)

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
