from __future__ import division

import random
import copy
import numpy as np

from sft.agent.DeepQAgentReplay import DeepQAgentReplay


class DeepQAgentReplayCloning(DeepQAgentReplay):
	# actions: possible actions
	# gamma: discount factor
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# model: estimator for q values
	# batch size: size of minibatches for experience replay (see doi:10.1038/nature14236)
	# buffer size: size of experience pool from which minibatches are randomly sampled
	# start_learn: after how many experiences (buffer size) we start learning based on experiences
	# steps_clone: number of steps we clone the network after
	def __init__(self, logger, actions, discount, model, batch_size, buffer_size, start_learn, steps_clone, learn_steps):
		super(DeepQAgentReplayCloning, self).__init__(logger, actions, discount, model, batch_size, buffer_size, start_learn,
													  learn_steps)
		self.steps_clone = steps_clone
		self.steps_clone_count = 0
		self.model_cloned = model.clone()

	def incorporate_reward(self, old_state, action, new_state, reward):
		super(DeepQAgentReplayCloning, self).incorporate_reward(old_state, action, new_state, reward)
		# increment step count for model cloning
		self.steps_clone_count += 1
		if self.steps_clone_count == self.steps_clone:
			# self.logger.log_message("Clone model")
			# transfer weights from current model to cloned one
			self.model_cloned.copy_from(self.model)
			self.steps_clone_count = 0

	def _get_target(self, old_state, action, new_state, reward):
		# re-predict qs values for old state from current model
		qs_old = self.model.predict_qs(old_state)
		# create target vector for this training sample
		target_qs = np.zeros((len(self.actions, )))
		# copy over target qs values predicted by current model
		# this will lead to a mean-squared error of 0 when fitting the model with these target value
		target_qs[:] = qs_old[:]
		if new_state is not None:
			# use cloned model to predict q values for new state
			qs_new = self.model_cloned.predict_qs(new_state)
			# get the max q-value (corresponding to the 'best' future action)
			q_max_new = np.max(qs_new)
			# calculate target q-value for selected action based on reward and discounted maximal future q-value
			target_q_ai = reward + (self.gamma * q_max_new)
		else:  # terminal
			# when new state is terminal just use reward as target q-value for selected action
			target_q_ai = reward
		# index of selected action
		ai = self.actions.index(action)
		# in target vector overwrite target q-value for selected action
		target_qs[ai] = target_q_ai
		return target_qs
