import random
from lat.RobotAgent import RobotAgent
import numpy as np


class DeepQAgent(RobotAgent):
	_qs_old = None
	_qs_old_state = None

	# actions: possible actions
	# gamma: discount factor
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# model: estimator for q values
	def __init__(self, actions, gamma, epsilon, epsilon_update, model):
		self._actions = actions
		self._gamma = gamma
		self._epsilon = epsilon
		self._epsilon_update = epsilon_update
		self._model = model

	def choose_action(self, curr_state):
		qs = self._model.predict_qs(curr_state)
		# store qs for current state because usually we can use them in subsequent call to incorporate_reward
		self._qs_old = qs
		self._qs_old_state = curr_state
		if random.random() < self._epsilon:
			ai = np.random.randint(0, len(self._actions))
			# print("Chosen action randomly")
		else:
			ai = np.argmax(qs)
			# print("Chosen action based on qs: {0}".format(qs))
		return self._actions[ai]

	def incorporate_reward(self, old_state, action, new_state, reward):
		# retrieved stored qs for old state or re-predict them
		qs_old = self._qs_old if np.array_equal(self._qs_old_state, old_state) else self._model.predict_qs(old_state)
		target_qs = np.zeros((1, len(self._actions)))
		target_qs[:] = qs_old[:]
		qs_new = self._model.predict_qs(new_state)
		q_max_new = np.max(qs_new)
		if reward == -1:  # non-terminal
			update = reward + (self._gamma * q_max_new)
		else:  # terminal
			update = reward
		ai = self._actions.index(action)
		target_qs[0, ai] = update
		self._model.update_qs(old_state, target_qs)
		# print("Incorporated reward")

	def new_epoch(self):
		self._epsilon = self._epsilon_update(self._epsilon)
		# print("Updated epsilon: {0}".format(self._epsilon))
