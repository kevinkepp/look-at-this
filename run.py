import random
import numpy as np
from lat.DeepQAgentReplay import DeepQAgentReplay
from lat.SimpleReward import RewardAtTheEnd, LinearReward
from lat.DeepQAgent import DeepQAgent
from lat.Evaluator import Evaluator
from lat.QAgent import QAgent
from lat.RandomAgent import RandomAgent
# from lat.OldSimulator import Simulator, Actions
from lat.Simulator import SimpleMatrixSimulator as Simulator, Actions
from lat.KerasMlpModel import KerasMlpModel

## Global parameters
EPOCHS = 500  # runs/games
GRID_SIZE_N = 7
GRID_SIZE_M = GRID_SIZE_N
MAX_STEPS = GRID_SIZE_N * GRID_SIZE_M  # max steps per run/game
BOUNDED = False  # false means terminate on out of bounds, true means no out of bounds possible

## Environment parameters
ACTIONS = Actions.all()
# different reward functions
REWARD_LIN = LinearReward()
REWARD_AT_END = RewardAtTheEnd()

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
EPSILON_UPDATE_ATARI_SMALL = lambda n: _EPSILON_UPDATE_LIN_FAST(n, min(EPOCHS / 20, 100))
# multiply by fixed factor but minimum min_e
_EPSILON_UPDATE_FACTOR = lambda n, f: max(np.power(f, n), EPSILON_MIN)
# custom update function with convex shape, adapted -log(x) with f(1) = 1 and f(EPOCHS) = 0
def EPSILON_UPDATE_KEV(n):
	a = (np.exp(-1) * EPOCHS - 1) / (1 - np.exp(-1))
	return max(-np.log((n + a) / (EPOCHS + a)), EPSILON_MIN)
# q value init (QAgent)
Q_INIT = lambda state, action: random.random()
# MLP architecture (DeepQAgent)
MODEL_IN_LAYER_SIZE = GRID_SIZE_N * GRID_SIZE_M  # input layer size
MODEL_HID_LAYER_SIZES = []  # hidden layers and sizes, for now defaults to no hidden layer
MODEL_OUT_LAYER_SIZE = len(ACTIONS)  # output layer size
# replay batch size (DeepQAgentReplay)
REPLAY_BATCH_SIZE_ATARI = 32
# replay buffer (DeepQAgentReplay)
REPLAY_BUFFER_ATARI = EPOCHS / 50  # (see doi:10.1038/nature14236)
REPLAY_BUFFER_ATARI_SMALL = min(EPOCHS / 20, 100)  # like atari but for smaller EPOCHS

## Setup agents and environments
envs = []
names = []
# Random Agent as baseline
agent = RandomAgent(ACTIONS)
env = Simulator(agent, REWARD_LIN, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("Random")
# Q-Agent
agent = QAgent(ACTIONS, ALPHA, GAMMA, Q_INIT)
env = Simulator(agent, REWARD_LIN, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("QAgent")
# Deep Q-Agents
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model,
						 REPLAY_BATCH_SIZE_ATARI, REPLAY_BUFFER_ATARI_SMALL)
env = Simulator(agent, REWARD_LIN, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[] g=" + str(GAMMA))

model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, 0.9, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model,
						 REPLAY_BATCH_SIZE_ATARI, REPLAY_BUFFER_ATARI_SMALL)
env = Simulator(agent, REWARD_LIN, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[] g=0.9")

model = KerasMlpModel(MODEL_IN_LAYER_SIZE, [50], MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model,
						 REPLAY_BATCH_SIZE_ATARI, REPLAY_BUFFER_ATARI_SMALL)
env = Simulator(agent, REWARD_LIN, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[50] g=" + str(GAMMA))

model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, 0.9, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model,
						 REPLAY_BATCH_SIZE_ATARI, REPLAY_BUFFER_ATARI_SMALL)
env = Simulator(agent, REWARD_LIN, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[50] g=0.9")

model = KerasMlpModel(MODEL_IN_LAYER_SIZE, [150, 75], MODEL_OUT_LAYER_SIZE)
agent = DeepQAgent(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE_ATARI_SMALL, model)
env = Simulator(agent, REWARD_LIN, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, bounded=BOUNDED)
envs.append(env)
names.append("DeepQAgent[150, 75]")

## Evaluate results
# choose which agents to run
include = [2, 4]
envs = [envs[i] for i in include]
names = [names[i] for i in include]
# run and evaluate agents
ev = Evaluator(envs, names, EPOCHS, grid="{0}x{1}".format(GRID_SIZE_N, GRID_SIZE_M), gamma=GAMMA)
ev.run(True)

# hacky way of visualizing the weights
if False:
	import matplotlib.pyplot as plt

	f = plt.figure()
	# hacky access to weights
	weights = envs[1]._agent._model._model.get_weights()[0]
	# print("Weights for " + names[1] + ":\n" + str(weights))
	for i in range(weights.shape[1]):
		ax = f.add_subplot(2, 2, i + 1)
		w = weights[:, i].reshape((GRID_SIZE_N, GRID_SIZE_M))
		ax.imshow(w, interpolation='nearest', aspect='auto', cmap="Blues")
		ax.set_title("Action " + str([a for a in Actions][i].name))
	plt.show()
