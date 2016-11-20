from __future__ import division
from lat.DeepQAgentReplay import DeepQAgentReplay
import numpy as np


class DeepQAgentPositiveReplay(DeepQAgentReplay):

	DEF_BATCH_SIZE = 8
	DEF_BUFFER = 200
	DEF_POS_PORTION = 0.6
	DEF_EPSILON_POS_RPL = 0.1

	def __init__(self, actions, gamma, epsilon, epsilon_update, model, batch_size=DEF_BATCH_SIZE, buffer=DEF_BUFFER, pos_portion=DEF_POS_PORTION, epsilon_pos_rpl=DEF_EPSILON_POS_RPL):
		super(DeepQAgentPositiveReplay, self).__init__(actions, gamma, epsilon, epsilon_update, model, batch_size, buffer)
		self.len_repl_pos = int(np.round(pos_portion * buffer,0))
		self.len_repl_non_pos = int(np.round((1-pos_portion) * buffer,0))
		self.buffer = buffer
		self.replay = []
		self.replay_pos = []
		self.replay_non_pos = []
		self.epsilon_pos_repl = epsilon_pos_rpl
		assert self.buffer == self.len_repl_non_pos +  self.len_repl_pos

	def _update_replay_list(self, exp_new):
		""" used to update the replay list with new experiences """
		# exp_new = (old_state, action, new_state, reward)
		reward = exp_new[3]
		if reward > 0:
			self._update_positive_replay(exp_new)
		else:
			self._update_non_pos_replay(exp_new)

	def _update_positive_replay(self, exp_new):
		""" used to update the positive replay list """
		if self.len_repl_pos > len(self.replay_pos):
			self.replay_pos.append(exp_new)
		else:
			# really random throw away only in self.epsilon_pos_repl of the cases
			i_arr = range(self.len_repl_pos)
			if np.random.binomial(1, self.epsilon_pos_repl):
				p = None
			else:
				np_pos_rpl = np.array(self.replay_pos)
				p = np.float64(1. / np_pos_rpl[:, 3])
				p = p / p.sum()
				#print(p.sum())
			i = np.random.choice(i_arr, p=p)
			self.replay_pos[i] = exp_new
		self.replay = self.replay_pos + self.replay_non_pos

	def _update_non_pos_replay(self, exp_new):
		""" used to update the non positive replay list """
		if self.len_repl_non_pos > len(self.replay_non_pos):
			self.replay_non_pos.append(exp_new)
		else:
			i_arr = range(self.len_repl_non_pos)
			i = np.random.choice(i_arr)
			self.replay_non_pos[i] = exp_new
		self.replay = self.replay_pos + self.replay_non_pos
