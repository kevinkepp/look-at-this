from __future__ import division

import random

import numpy as np

from sft.agent.DeepQAgentReplayCloning import DeepQAgentReplayCloning


class ReplayList:
	def __init__(self, max_len):
		self.max_len = max_len
		self.exp_list = []
		self.curr_i = 0


class DeepQAgentPosPathReplay(DeepQAgentReplayCloning):
	# portions of: i=0 -> positive reward experiences, i=1 -> experiences that preceded positive rewards, i=2 -> other experiences
	DEF_PORTIONS = [0.05, 0.5, 0.45]

	def __init__(self, logger, actions, discount, model, batch_size, buffer_size, start_learn, steps_clone, portions=DEF_PORTIONS):
		super(DeepQAgentPosPathReplay, self).__init__(logger, actions, discount, model, batch_size, buffer_size, start_learn, steps_clone)
		assert np.sum(portions) == 1
		self.portions = portions
		max_len_pos = int(np.round(portions[0] * buffer_size, 0))
		max_len_path_to_pos = int(np.round(portions[1] * buffer_size, 0))
		max_len_others = int(np.round(portions[2] * buffer_size, 0))
		self.rpl_pos = ReplayList(max_len_pos)
		self.rpl_path_to_pos = ReplayList(max_len_path_to_pos)
		self.rpl_others = ReplayList(max_len_others)
		self.last_exps = []
		assert self.buffer == self.rpl_pos.max_len + self.rpl_path_to_pos.max_len + self.rpl_others.max_len

	def _get_mini_batch(self):
		""" overwrites fct, to get a customized mini batch """
		mini_len_pos = int(np.round(self.batch_size * self.portions[0], 0))
		mini_len_path = int(np.round(self.batch_size * self.portions[1], 0))
		mini_len_others = int(np.round(self.batch_size * self.portions[2], 0))
		assert mini_len_pos + mini_len_path + mini_len_others == self.batch_size
		batch_pos = self._get_one_batch(self.rpl_pos.exp_list, mini_len_pos)
		batch_path = self._get_one_batch(self.rpl_path_to_pos.exp_list, mini_len_path)
		batch_oth = self._get_one_batch(self.rpl_others.exp_list, mini_len_others)
		return batch_pos + batch_path + batch_oth

	def _get_one_batch(self, repllist, sample_len):
		if len(repllist) < sample_len:
			return repllist
		else:
			return random.sample(repllist, sample_len)

	def new_episode(self):
		# nothing needs to happen when a new episode starts in here
		# exp_new = (old_state, action, new_state, reward)
		if len(self.last_exps) > 0:
			very_last_exp = self.last_exps[-1]
			reward = very_last_exp[3]
			if reward > 0:
				self._update_with_one_exp(self.rpl_pos, very_last_exp)
				self._update_a_rpllist(self.rpl_path_to_pos, self.last_exps)
			else:
				self._update_a_rpllist(self.rpl_others, self.last_exps)
			self.last_exps = []
			param = "{}\t{}\t{}\t\t{}\t{}\t{}".format(len(self.rpl_pos.exp_list), len(self.rpl_path_to_pos.exp_list), len(self.rpl_others.exp_list), self.rpl_pos.curr_i, self.rpl_path_to_pos.curr_i, self.rpl_others.curr_i)
			self.logger.log_parameter("replay_lists_lengths", param)
			self.replay = self.rpl_pos.exp_list + self.rpl_path_to_pos.exp_list + self.rpl_others.exp_list

	def _update_replay_list(self, exp_new):
		""" used to update the replay list with new experiences """
		self.last_exps.append(exp_new)

	def _update_a_rpllist(self, replaylist, exps):
		""" update a entire list with a list of exps """
		for exp in exps:
			self._update_with_one_exp(replaylist, exp)

	def _update_with_one_exp(self, replaylist, exp):
		""" update a list with one experience """
		if replaylist.max_len > len(replaylist.exp_list):
			replaylist.exp_list.append(exp)
		else:
			replaylist.exp_list[replaylist.curr_i] = exp
			replaylist.curr_i += 1
			if replaylist.curr_i == replaylist.max_len:
				replaylist.curr_i = 0
