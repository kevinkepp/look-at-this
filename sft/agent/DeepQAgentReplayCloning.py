from __future__ import division

import random
import copy
import numpy as np

from sft.agent.DeepQAgent import DeepQAgent
from sft.agent.DeepQAgentReplay import DeepQAgentReplay


# TODO this might not work yet, look at papers again!
class DeepQAgentReplayCloning(DeepQAgentReplay):

	# actions: possible actions
	# gamma: discount factor
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# model: estimator for q values
	# batch size: size of minibatches for experience replay (see doi:10.1038/nature14236)
	# buffer size: size of experience pool from which minibatches are randomly sampled
	# k: number of steps we clone the network after
	def __init__(self, actions, discount, model, batch_size, buffer_size, k):
		super(DeepQAgentReplayCloning, self).__init__(actions, discount, model, batch_size, buffer_size)
		self.k = k
		self.model_cloned = copy.deepcopy(model)
		self.k_count = 0

	def incorporate_reward(self, old_state, action, new_state, reward):
		print("Reward " + str(reward))
		""" incorporates reward, states, action into replay list and updates the parameters of model """
		exp_new = (old_state, action, new_state, reward)
		self._update_replay_list(exp_new)
		if len(self.replay) == self.buffer:
			mini_batch = random.sample(self.replay, self.batch_size)
			self._update_parameters_with_batch(mini_batch)
		self.k_count += 1
		if self.k_count == self.k:
			print("Clone")
			self.model_cloned = copy.deepcopy(self.model)
			self.k_count = 0

	def _update_parameters_with_batch(self, mini_batch):
		""" used to update model parameters with the chosen batch """
		states_train = []
		targets_train = []
		for old_state, action, new_state, reward in mini_batch:
			# re-predict qs values for old state
			qs_old = self.model_cloned.predict_qs(old_state)
			target_qs = np.zeros((len(self.actions,)))
			target_qs[:] = qs_old[:]
			if new_state is not None:
				# use cloned model to predict qs for new state
				qs_new = self.model_cloned.predict_qs(new_state)
				q_max_new = np.max(qs_new)
				update = reward + (self.gamma * q_max_new)
			else:  # terminal
				update = reward
			ai = self.actions.index(action)
			target_qs[ai] = update
			states_train.append(old_state)
			targets_train.append(target_qs)
		# train with experience batch
		states_train = np.array(states_train)
		targets_train = np.array(targets_train)
		self.model.update_qs(states_train, targets_train)

	def _update_replay_list(self, exp_new):
		""" used to update the replay list with new experiences """
		if len(self.replay) < self.buffer:
			self.replay.append(exp_new)
		else:
			# buffer is full, from now on overwrite old values
			self.h = self.h + 1 if self.h < self.buffer - 1 else 0
			self.replay[self.h] = exp_new
