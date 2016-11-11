import random

from lat.DeepQAgent import DeepQAgent
import numpy as np


class DeepQAgentReplay(DeepQAgent):
	DEF_BATCH_SIZE = 32
	DEF_BUFFER = 100

	# actions: possible actions
	# gamma: discount factor
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# model: estimator for q values
	# batch size: size of minibatches for experience replay (see doi:10.1038/nature14236)
	# buffer size: size of experience pool from which minibatches are randomly sampled
	def __init__(self, actions, gamma, epsilon, epsilon_update, model, batch_size=DEF_BATCH_SIZE, buffer=DEF_BUFFER):
		super(DeepQAgentReplay, self).__init__(actions, gamma, epsilon, epsilon_update, model)
		self.batch_size = batch_size
		self.buffer = max(buffer, batch_size)  # buffer has to be at least batch_size
		self.replay = []
		self.h = 0

	def incorporate_reward(self, old_state, action, new_state, reward):
		exp_new = (old_state, action, new_state, reward)
		if len(self.replay) < self.buffer:
			self.replay.append(exp_new)
		else:
			# buffer is full, from now on overwrite old values
			self.h = self.h + 1 if self.h < self.buffer - 1 else 0
			self.replay[self.h] = exp_new
			mini_batch = random.sample(self.replay, self.batch_size)
			states_train = []
			targets_train = []
			for old_state, action, new_state, reward in mini_batch:
				# re-predict qs values for old state
				qs_old = self.model.predict_qs(old_state)
				target_qs = np.zeros((len(self.actions,)))
				target_qs[:] = qs_old[:]
				if new_state is not None:
					qs_new = self.model.predict_qs(new_state)
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
		# print("Incorporated reward")