from __future__ import division

import random

import numpy as np

from sft.agent.RobotAgent import RobotAgent


class DeepQAgent(RobotAgent):
	qs_old = None
	qs_old_state = None

	# actions: possible actions
	# gamma: discount factor
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# model: estimator for q values
	def __init__(self, logger, actions, gamma, model):
		self.logger = logger
		self.actions = actions
		self.gamma = gamma
		self.model = model

	def choose_action(self, curr_state, eps):
		qs = self.model.predict_qs(curr_state)
		self.logger.log_parameter("q", qs)
		# print("Q-Values " + str(qs))
		# store qs for current state because usually we can use them in subsequent call to incorporate_reward
		self.qs_old = qs
		self.qs_old_state = curr_state
		if random.random() < eps:
			ai = np.random.randint(0, len(self.actions))
		else:
			ai = np.argmax(qs)
			# print("Chosen action based on qs: {0}".format(qs))
		return self.actions[ai]

	def incorporate_reward(self, old_state, action, new_state, reward):
		# retrieved stored qs for old state or re-predict them
		qs_old = self.qs_old if np.array_equal(self.qs_old_state, old_state) else self.model.predict_qs(old_state)
		target_qs = np.zeros((len(self.actions),))
		target_qs[:] = qs_old[:]
		if new_state is not None:
			qs_new = self.model.predict_qs(new_state)
			q_max_new = np.max(qs_new)
			update = reward + (self.gamma * q_max_new)
		else:  # terminal
			update = reward
		ai = self.actions.index(action)
		target_qs[ai] = update
		self.model.update_qs(old_state, target_qs)
		# print("Incorporated reward")
