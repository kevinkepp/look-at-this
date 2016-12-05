from __future__ import division
import random
import numpy as np

from lat.PathSimulator import PathSimulator, PathSimulatorSimple, PathSimSimpleExpansiveSampler, PathSimExpSplImages
from lat.RandomAgent import RandomAgent
from lat.QAgent import QAgent
from lat.KerasMlpModel import KerasMlpModel
from lat.DeepQAgent import DeepQAgent
from lat.DeepQAgentReplay import DeepQAgentReplay
from lat.DeepQAgentPositiveRepay import DeepQAgentPositiveReplay
from lat.SimpleReward import RewardAtTheEnd, LinearReward, MiddleAsReward, MiddleAbsoluteAsReward, SumAsReward, RewardAtTheEndForOneInTheMiddle
from lat.OldSimulator import Simulator as OldSimulator, Actions as OldActions
from lat.SimpleVisualize import PathAndResultsPlotter
from lat.Simulator import SimpleMatrixSimulator, GaussSimulator, ImageSimulator, ImageSimulatorSpecialSample, Actions
from lat.Evaluator import Evaluator

## Global parameters
EPOCHS = 3000  # runs/games
GRID_SIZE_N = 15
GRID_SIZE_M = 15
WORLD_SIZE_FACTOR = 10
# max steps per run/game
MAX_STEPS = GRID_SIZE_N * WORLD_SIZE_FACTOR * 10
BOUNDED = False  # false means terminate on out of bounds, true means no out of bounds possible

## Environment parameters
SIMULATOR = PathSimExpSplImages #PathSimSimpleExpansiveSampler #PathSimulatorSimple  # ImageSimulator  # ImageSimulatorSpecialSample
SIMULATOR_NAME = "PathSimExpSplImages"
# ACTIONS = Actions.all()
ACTIONS = Actions.all
# different reward functions
REWARD_LIN = LinearReward()
REWARD_LIN_NAME = "linear"
REWARD_AT_END = RewardAtTheEnd()  # also called "constant"
REWARD_AT_END_NAME = "at-end"
REWARD_AT_END_MIDDLE = RewardAtTheEndForOneInTheMiddle()
REWARD_AT_END_NAME = "at-end-if-middle-one"
REWARD_MID = MiddleAsReward()
REWARD_MID_NAME = "middle"
REWARD_MID_ABS = MiddleAbsoluteAsReward()
REWARD_MID_ABS_NAME = "middle-abs"
REWARD_SUM = SumAsReward()
REWARD_SUM_NAME = "sum"
# actual reward we want to use
REWARD = REWARD_AT_END
REWARD_NAME = REWARD_AT_END_NAME

## Agent parameters
# learning rate for Q-Agents
ALPHA = 1  # 1 is optimal for deterministic env
# discount factor
GAMMA = 0.99  # (see doi:10.1038/nature14236)
# epsilon-greedy strategy: agent chooses random action if rand() < epsilon
EPSILON_START = 1
# minimum epsilon in order to always guarantee a random influence (see doi:10.1038/nature14236)
EPSILON_MIN = 0.05
# linear decrease over number of epochs but minimum min_e
EPSILON_UPDATE_LIN = lambda n: max(1 - n / EPOCHS, EPSILON_MIN)
# linear decrease to n_min
_EPSILON_UPDATE_LIN_UNTIL = lambda n, n_min: max(1 - n / n_min, EPSILON_MIN)
# linear decrease to epochs/50, works for large EPOCHS (see doi:10.1038/nature14236)
EPSILON_UPDATE_ATARI = lambda n: _EPSILON_UPDATE_LIN_UNTIL(n, EPOCHS / 50)
# linear decrease to epochs/20, like atari but for smaller EPOCHS
EPSILON_UPDATE_ATARI_SMALL = lambda n: _EPSILON_UPDATE_LIN_UNTIL(n, max(EPOCHS / 4, 100))
# multiply by fixed factor but minimum min_e
_EPSILON_UPDATE_FACTOR = lambda n, f: max(np.power(f, n), EPSILON_MIN)
# custom update function with convex shape, adapted -log(x) with f(1) = 1 and f(EPOCHS) = 0
def EPSILON_UPDATE_KEV(n):
	a = (np.exp(-1) * EPOCHS - 1) / (1 - np.exp(-1))
	return max(-np.log((n + a) / (EPOCHS + a)), EPSILON_MIN)
# actual epsilon-greedy strategy we want to use
EPSILON_UPDATE = EPSILON_UPDATE_LIN  # EPSILON_UPDATE_ATARI_SMALL
EPSILON_UPDATE_NAME = "linear"  # "atari-small"
# q value init (QAgent)
Q_INIT = lambda state, action: random.random()
# MLP architecture (DeepQAgent)
MODEL_IN_LAYER_SIZE = GRID_SIZE_N * GRID_SIZE_M  # input layer size
MODEL_HID_LAYER_SIZES = []  # hidden layers and sizes, for now defaults to no hidden layer
MODEL_OUT_LAYER_SIZE = len(ACTIONS)  # output layer size
# replay batch size (DeepQAgentReplay)
REPLAY_BATCH_SIZE_ATARI = 32
REPLAY_BATCH_SIZE_ATARI_SMALL = 8
REPLAY_BATCH_SIZE = REPLAY_BATCH_SIZE_ATARI_SMALL
# replay buffer (DeepQAgentReplay)
REPLAY_BUFFER_ATARI = EPOCHS / 50  # (see doi:10.1038/nature14236)
REPLAY_BUFFER_ATARI_SMALL = max(EPOCHS / 10, 100)  # like atari but for smaller EPOCHS
# actual replay buffer size we want to use
REPLAY_BUFFER = REPLAY_BUFFER_ATARI_SMALL

## Setup agents and environments
envs = []
names = []

# IMG_PATH = "tmp/white_circle.png"  # "tmp/moon.jpg"
VISUALIZER = PathAndResultsPlotter()
def create_simulator(agent):
	return SIMULATOR(agent, REWARD, GRID_SIZE_N, GRID_SIZE_M, max_steps=MAX_STEPS, visualizer=VISUALIZER,
					 bounded=BOUNDED, world_size_factor=10)

# Random Agent as baseline
agent = RandomAgent(ACTIONS)
envs.append(create_simulator(agent))
names.append("Random")

# Q-Agent
agent = QAgent(ACTIONS, ALPHA, 0.8, Q_INIT)
envs.append(create_simulator(agent))
names.append("QAgent g=0.8")

# Deep Q-Agents
model = KerasMlpModel(MODEL_IN_LAYER_SIZE, MODEL_HID_LAYER_SIZES, MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE, model, REPLAY_BATCH_SIZE, REPLAY_BUFFER)
envs.append(create_simulator(agent))
names.append("DeepQAgentReplay[]")

model = KerasMlpModel(MODEL_IN_LAYER_SIZE, [8], MODEL_OUT_LAYER_SIZE)
agent = DeepQAgentReplay(ACTIONS, GAMMA, EPSILON_START, EPSILON_UPDATE, model, REPLAY_BATCH_SIZE, REPLAY_BUFFER)
envs.append(create_simulator(agent))
names.append("DeepQAgentReplay[8]")

## Evaluate results
# choose which agents to run
include = [2]
envs = [envs[i] for i in include]
names = [names[i] for i in include]
# run and evaluate agents
ev = Evaluator(envs, names, EPOCHS, VISUALIZER, grid="{0}x{1}".format(GRID_SIZE_N, GRID_SIZE_M), actions=len(ACTIONS),
			   simulator=SIMULATOR_NAME, max_steps=MAX_STEPS, discount=GAMMA, reward=REWARD_NAME,
			   eps_min=EPSILON_MIN, eps_update=EPSILON_UPDATE_NAME)  # img=IMG_PATH.split("/")[-1])
# TODO
ev.run(None, True)
# ev.run(lambda score: score < GRID_SIZE_N * 0.002, True)

# hacky way of visualizing the weights
# TODO improve and include this in Evaluator
if False:
	import matplotlib.pyplot as plt

	f = plt.figure()
	agent_i = 0
	# hacky access to weights
	agent = envs[agent_i].agent
	weights = agent.model._model.layers[0].get_weights()[0]
	# print(weights.shape)
	# print("Weights for " + names[1] + ":\n" + str(weights))
	for i in range(weights.shape[1]):
		ax = f.add_subplot(2, 2, i + 1)
		w = weights[:, i].reshape((GRID_SIZE_N, GRID_SIZE_M))
		ax.imshow(w, interpolation='nearest', aspect='auto', cmap="Blues")
		# (up=0, down=1, left=2, right=3)
		action_dict = {0: "up", 1: "down", 2: "left", 3: "right"}
		ax.set_title("Action " + action_dict[i])
	plt.show()
	plt.savefig("tmp_weights_" + names[agent_i] + ".png")