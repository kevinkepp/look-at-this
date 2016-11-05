import random
from lat.RobotAgent import RobotAgent
import numpy as np


class DeepQAgent(RobotAgent):
	_qs_old = None
	_qs_old_state = None

	# gamma: discount factor
	# epsilon: epsilon-greedy strategy
	# model: estimator for q values
	def __init__(self, gamma, epsilon, epsilon_update, actions_count, model):
		self._gamma = gamma
		self._epsilon = epsilon
		self._epsilon_update = epsilon_update
		self._actions_count = actions_count
		self._model = model

	def choose_action(self, curr_state):
		qs = self._model.predict_qs(curr_state)
		# store qs for current state because usually we can use them in subsequent call to incorporate_reward
		self._qs_old = qs
		self._qs_old_state = curr_state
		if random.random() < self._epsilon:
			action = np.random.randint(0, self._actions_count)
			print("Chosen action randomly")
		else:
			action = np.argmax(qs)
			print("Chosen action based on qs: " + str(qs))
		return action

	def incorporate_reward(self, old_state, action, new_state, reward):
		# retrieved stored qs for old state or re-predict them
		qs_old = self._qs_old if np.array_equal(self._qs_old_state, old_state) else self._model.predict_qs(old_state)
		target_qs = np.zeros((1, self._actions_count))
		target_qs[:] = qs_old[:]
		qs_new = self._model.predict_qs(new_state)
		q_max_new = np.max(qs_new)
		if reward == -1:  # non-terminal
			update = reward + (self._gamma * q_max_new)
		else:  # terminal
			update = reward
		target_qs[0, action] = update
		self._model.update_qs(old_state, target_qs)
		print("Incorporated reward")

	def new_epoch(self):
		self._epsilon = self._epsilon_update(self._epsilon)
		print("Updated epsilon: " + str(self._epsilon))
