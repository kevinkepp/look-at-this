import random
from lat.RobotAgent import RobotAgent
import numpy as np


class DeepQAgent(RobotAgent):
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
		if random.random() < self._epsilon:
			action = np.random.randint(0, self._actions_count)
			print("Chosen action randomly")
		else:
			action = np.argmax(qs)
			print("Chosen action based on qs: " + str(qs))
		return action

	def incorporate_reward(self, old_state, action, new_state, reward):
		qs = self._model.predict_qs(new_state)
		q_max = np.max(qs)
		target_qs = np.zeros((1, self._actions_count))
		target_qs[:] = qs[:]
		update = reward + (self._gamma * q_max)
		target_qs[0, action] = update
		self._model.update_qs(old_state, target_qs)
		print("Incorporated reward")

	def new_epoch(self):
		self._epsilon = self._epsilon_update(self._epsilon)
		print("Updated epsilon: " + str(self._epsilon))
