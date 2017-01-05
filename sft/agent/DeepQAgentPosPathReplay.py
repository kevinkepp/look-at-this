from __future__ import division

import numpy as np

from sft.agent.DeepQAgentReplay import DeepQAgentReplay


class ReplayList:
	def __init__(self, max_len):
		self.max_len = max_len
		self.exp_list = []
		self.curr_i = 0


class DeepQAgentPosPathReplay(DeepQAgentReplay):
	# portions of: i=0 -> positive reward experiences, i=1 -> experiences that preceded positive rewards, i=2 -> other experiences
	DEF_PORTIONS = [0.3, 0.5, 0.2]
	DEF_RANDOM_REPLACE = False
	DEF_LEN_EXP_TO_POS_REWARD = 150
	# DEF_EPSILON_POS_RPL = 0.1

	def __init__(self, logger, actions, discount, model, batch_size, buffer_size, start_learn, portions=DEF_PORTIONS, len_path_to_pos=DEF_LEN_EXP_TO_POS_REWARD):
		super(DeepQAgentPosPathReplay, self).__init__(logger, actions, discount, model, batch_size, buffer_size, start_learn)
		assert np.sum(portions) == 1
		max_len_pos = int(np.round(portions[0] * buffer_size, 0))
		max_len_path_to_pos = int(np.round(portions[1] * buffer_size, 0))
		max_len_others = int(np.round(portions[2] * buffer_size, 0))
		self.rpl_pos = ReplayList(max_len_pos)
		self.rpl_path_to_pos = ReplayList(max_len_path_to_pos)
		self.rpl_others = ReplayList(max_len_others)
		self.len_path_to_pos = len_path_to_pos
		self.last_exps = []
		assert self.buffer == self.rpl_pos.max_len + self.rpl_path_to_pos.max_len + self.rpl_others.max_len

	def _update_replay_list(self, exp_new):
		""" used to update the replay list with new experiences """
		# TODO currently workarround to use meaningful paths by saving the last 150 steps, but if episode is shorter, also the steps before that episode are included
		# TODO best way would be to signal the agent, that an episode is over
		# exp_new = (old_state, action, new_state, reward)
		reward = exp_new[3]
		if reward > 0:
			self._update_with_one_exp(self.rpl_pos, exp_new)
			self._update_a_rpllist(self.rpl_path_to_pos, self.last_exps[-self.len_path_to_pos:])
			self.last_exps = []
		else:
			self.last_exps.append(exp_new)
			self._update_with_one_exp(self.rpl_others, exp_new)
		param = "{}\t{}\t{}".format(len(self.rpl_pos.exp_list), len(self.rpl_path_to_pos.exp_list), len(self.rpl_others.exp_list))
		self.logger.log_parameter("replay_lists_lengths", param)
		self.replay = self.rpl_pos.exp_list + self.rpl_path_to_pos.exp_list + self.rpl_others.exp_list

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
