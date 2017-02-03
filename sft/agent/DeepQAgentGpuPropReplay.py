from __future__ import division

import numpy as np
import theano

from sft.agent.DeepQAgentGpu import ReplayBuffer, DeepQAgentGpu


class PropReplayBuffer(object):
	def __init__(self, size, view_size, action_hist_size, pct_pos):
		self.size = size
		self.v = np.zeros((size, 1, view_size.w, view_size.h), dtype=theano.config.floatX)
		self.ah = np.zeros((size, 1, action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX)
		self.a = np.zeros((size, 1), dtype=np.int32)
		self.v2 = np.zeros((size, 1, view_size.w, view_size.h), dtype=theano.config.floatX)
		self.ah2 = np.zeros((size, 1, action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX)
		self.r = np.zeros((size, 1), dtype=theano.config.floatX)
		self.t = np.zeros((size, 1), dtype=np.int32)
		# for prop rpl
		max_len_pos = int(np.round(pct_pos * size, 0))
		self.pct_pos = pct_pos
		self.top_pos = 0
		self.len_pos = 0
		self.start_oth = max_len_pos
		self.top_oth = max_len_pos
		self.len_oth = 0

	def add_to_pos(self, v, ah, a, v2, ah2, r, t):
		# write to current pos index
		self.v[self.top_pos] = v
		self.ah[self.top_pos] = ah
		self.a[self.top_pos] = a
		self.v2[self.top_pos] = v2
		self.ah2[self.top_pos] = ah2
		self.r[self.top_pos] = r
		self.t[self.top_pos] = t
		self.top_pos += 1
		if self.len_pos < self.start_oth:
			self.len_pos += 1
		if self.top_pos == self.start_oth:
			self.top_pos = 0

	def add_to_oth(self, v, ah, a, v2, ah2, r, t):
		# write to current other index
		self.v[self.top_oth] = v
		self.ah[self.top_oth] = ah
		self.a[self.top_oth] = a
		self.v2[self.top_oth] = v2
		self.ah2[self.top_oth] = ah2
		self.r[self.top_oth] = r
		self.t[self.top_oth] = t
		self.top_oth += 1
		if self.len_oth < self.size - self.start_oth:
			self.len_oth += 1
		if self.top_oth == self.size:
			self.top_oth = self.start_oth

	def _get_batch_indices(self, start, curr_len, batch_size):
		if curr_len > 0:
			return np.random.randint(start, start + curr_len, batch_size if curr_len > batch_size else curr_len)
		else:
			return None

	def draw_batch(self, batch_size):
		batch_size_pos = int(np.round(self.pct_pos * batch_size, 0))
		batch_size_oth = batch_size - batch_size_pos
		indices_pos = self._get_batch_indices(0, self.len_pos, batch_size_pos)
		indices_oth = self._get_batch_indices(self.start_oth, self.len_oth, batch_size_oth)
		if indices_pos is None:
			indices = indices_oth
		elif indices_oth is None:
			indices = indices_pos
		else:
			indices = np.append(indices_pos, indices_oth)
		return self.v[indices], self.ah[indices], self.a[indices], self.v2[indices], self.ah2[indices], \
			   self.r[indices], self.t[indices]


class DeepQAgentGpuPropReplay(DeepQAgentGpu):
	# actions: possible actions
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# batch size: size of minibatches for experience replay (see doi:10.1038/nature14236)
	# buffer size: size of experience pool from which minibatches are randomly sampled
	# start_learn: after how many experiences (buffer size) we start learning based on experiences
	# learn_steps: every learn_steps steps the model gets updated
	DEF_POS_PORTION = 0.5
	def __init__(self, logger, actions, batch_size, buffer_size, start_learn, learn_interval, view_size,
				 action_hist, model, pos_portion=DEF_POS_PORTION):
		self.prop_rpl_buffer = PropReplayBuffer(buffer_size, view_size, action_hist.get_size(), pos_portion)
		self.last_exps = []
		super(DeepQAgentGpuPropReplay, self).__init__(logger, actions, batch_size, buffer_size, start_learn, learn_interval, view_size,
				 action_hist, model)

	def incorporate_reward(self, old_state, action, new_state, reward):
		""" incorporates reward, states, action into replay list and updates the parameters of model """
		self.logger.log_parameter("reward", reward)
		old_view = old_state.view
		old_actions = self.action_hist.get_history(old_state.actions)
		is_terminal = new_state is None
		if not is_terminal:
			new_view = new_state.view
			new_actions = self.action_hist.get_history(new_state.actions)
		else:
			new_view = np.zeros(old_view.shape, dtype=theano.config.floatX)
			new_actions = np.zeros(old_actions.shape, dtype=theano.config.floatX)
		terminal = 1 if is_terminal else 0
		exp_new = (old_view, old_actions, action, new_view, new_actions, reward, terminal)

		self.last_exps.append(exp_new)

		len_all_rpl = self.prop_rpl_buffer.len_pos + self.prop_rpl_buffer.len_oth

		if len_all_rpl >= self.start_learn and self.learn_steps % self.learn_interval == 0:
			minibatch = self.prop_rpl_buffer.draw_batch(self.batch_size)
			self.model.update_qs(*minibatch)
			self.learn_steps = 1
		else:
			self.learn_steps += 1

	def new_episode(self):
		if len(self.last_exps)>0:
			final_reward = self.last_exps[-1][-2]
			self.logger.log_parameter("prop_rpl_list", "{}\t{}\t{}\t{}\t{}".format(self.prop_rpl_buffer.top_pos, self.prop_rpl_buffer.top_oth, self.prop_rpl_buffer.len_pos, self.prop_rpl_buffer.len_oth, self.prop_rpl_buffer.start_oth))
			l = len(self.last_exps)
			if final_reward == 1:
				for i in range(l):
					exp = self.last_exps[i]
					self.prop_rpl_buffer.add_to_pos(*exp)
			else:
				for i in range(l):
					exp = self.last_exps[i]
					self.prop_rpl_buffer.add_to_oth(*exp)
			self.last_exps = []
			self.action_hist.new_episode()
