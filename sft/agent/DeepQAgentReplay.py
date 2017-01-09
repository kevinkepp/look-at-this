from __future__ import division

import random

import numpy as np

from sft.agent.DeepQAgent import DeepQAgent


class DeepQAgentReplay(DeepQAgent):
	# actions: possible actions
	# gamma: discount factor
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# model: estimator for q values
	# batch size: size of minibatches for experience replay (see doi:10.1038/nature14236)
	# buffer size: size of experience pool from which minibatches are randomly sampled
	def __init__(self, logger, actions, discount, model, batch_size, buffer_size, start_learn, learn_steps):
		super(DeepQAgentReplay, self).__init__(logger, actions, discount, model)
		self.batch_size = batch_size
		self.buffer = max(buffer_size, batch_size)  # buffer has to be at least batch_size
		self.start_learn = start_learn
		self.learn_steps = learn_steps
		self.steps = 0
		self.replay = []
		self.h = 0

	def incorporate_reward(self, old_state, action, new_state, reward):
		""" incorporates reward, states, action into replay list and updates the parameters of model """
		# self.logger.log_parameter("reward", reward)
		exp_new = (old_state, action, new_state, reward)
		self._update_replay_list(exp_new)
		if len(self.replay) >= self.start_learn and self.steps % self.learn_steps == 0:
			mini_batch = random.sample(self.replay, self.batch_size)
			self._update_model(mini_batch)
			self.steps = 1
		else:
			self.steps += 1

	def _update_replay_list(self, exp_new):
		""" used to update the replay list with new experiences """
		if len(self.replay) < self.buffer:
			self.replay.append(exp_new)
		else:
			# buffer is full, from now on overwrite old values
			self.h = self.h + 1 if self.h < self.buffer - 1 else 0
			self.replay[self.h] = exp_new

	def _update_model(self, mini_batch):
		""" used to update model parameters with the chosen batch """
		features = []  # features
		targets = []  # targets
		for old_state, action, new_state, reward in mini_batch:
			features.append(old_state)
			target = self._get_target(old_state, action, new_state, reward)
			targets.append(target)
		# train with experience batch
		features = np.array(features)
		targets = np.array(targets)
		self.model.update_qs(features, targets)

	def _get_target(self, old_state, action, new_state, reward):
		# re-predict qs values for old state from current model
		qs_old = self.model.predict_qs(old_state)
		# create target vector for this training sample
		target_qs = np.zeros((len(self.actions, )))
		# copy over target qs values predicted by current model
		# this will lead to a mean-squared error of 0 when fitting the model with these target value
		target_qs[:] = qs_old[:]
		if new_state is not None:
			# predict q values for new state
			qs_new = self.model.predict_qs(new_state)
			# get the max q-value (corresponding to the 'best' future action)
			q_max_new = np.max(qs_new)
			# calculate target q-value for selected action based on reward and discounted maximal future q-value
			update = reward + (self.gamma * q_max_new)
		else:  # terminal
			# when new state is terminal just use reward as target q-value for selected action
			update = reward
		# index of selected action
		ai = self.actions.index(action)
		# in target vector overwrite target q-value for selected action
		target_qs[ai] = update
		return target_qs
