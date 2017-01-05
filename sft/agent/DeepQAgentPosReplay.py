from __future__ import division

import numpy as np

from sft.agent.DeepQAgentReplay import DeepQAgentReplay


class DeepQAgentPosReplay(DeepQAgentReplay):
	DEF_POS_PORTION = 0.5
	DEF_RANDOM_REPLACE = False
	# DEF_EPSILON_POS_RPL = 0.1

	def __init__(self, logger, actions, discount, model, batch_size, buffer_size, start_learn, pos_portion=DEF_POS_PORTION, random_replace=DEF_RANDOM_REPLACE):
		super(DeepQAgentPosReplay, self).__init__(logger, actions, discount, model, batch_size, buffer_size, start_learn)
		self.len_repl_pos = int(np.round(pos_portion * buffer_size, 0))
		self.len_repl_non_pos = int(np.round((1-pos_portion) * buffer_size, 0))
		self.replay_pos = []
		self.i_repl_pos = 0
		self.replay_non_pos = []
		self.i_repl_non_pos = 0
		self.random_replace = random_replace
		assert self.buffer == self.len_repl_non_pos + self.len_repl_pos

	def _update_replay_list(self, exp_new):
		""" used to update the replay list with new experiences """
		# exp_new = (old_state, action, new_state, reward)
		reward = exp_new[3]
		if reward > 0:
			self._update_positive_replay(exp_new)
		else:
			self._update_non_pos_replay(exp_new)
		self.replay = self.replay_pos + self.replay_non_pos

	def _update_positive_replay(self, exp_new):
		""" used to update the positive replay list """
		if self.random_replace:
			self._update_list_random(self.replay_pos, self.len_repl_pos, exp_new)
		else:
			self.i_repl_pos = self._update_list_sequential(self.replay_pos, self.len_repl_pos, exp_new, self.i_repl_pos)

	def _update_non_pos_replay(self, exp_new):
		""" used to update the non positive replay list """
		if self.random_replace:
			self._update_list_random(self.replay_non_pos, self.len_repl_non_pos, exp_new)
		else:
			self.i_repl_non_pos = self._update_list_sequential(self.replay_non_pos, self.len_repl_non_pos, exp_new, self.i_repl_non_pos)

	def _update_list_random(self, repl_list, len_repl_list, exp_new):
		""" used to update the replay lists with sequential choice of exp to be replaced"""
		if len_repl_list > len(repl_list):
			repl_list.append(exp_new)
		else:
			i_arr = range(len_repl_list)
			i = np.random.choice(i_arr)
			repl_list[i] = exp_new

	def _update_list_sequential(self, repl_list, len_repl_list, exp_new, curr_i):
		""" used to update the replay lists with random choice of exp to be replaced"""
		if len_repl_list > len(repl_list):
			repl_list.append(exp_new)
		else:
			repl_list[curr_i] = exp_new
			curr_i += 1
			if curr_i == len_repl_list:
				curr_i = 0
		return curr_i
