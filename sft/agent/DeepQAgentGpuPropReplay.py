from __future__ import division

import numpy as np
import theano

from sft.agent.DeepQAgentGpu import ReplayBuffer, DeepQAgentGpu


class DeepQAgentGpuPropReplay(DeepQAgentGpu):
	# actions: possible actions
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# batch size: size of minibatches for experience replay (see doi:10.1038/nature14236)
	# buffer size: size of experience pool from which minibatches are randomly sampled
	# start_learn: after how many experiences (buffer size) we start learning based on experiences
	# learn_steps: every learn_steps steps the model gets updated
	DEF_PORTIONS = [0.05, 0.6, 0.35]
	def __init__(self, logger, actions, batch_size, buffer_size, start_learn, learn_interval, view_size,
				 action_hist, model, portions=DEF_PORTIONS):
		assert np.sum(portions) == 1
		self.portions = portions
		max_len_pos = int(np.round(portions[0] * buffer_size, 0))
		max_len_path_to_pos = int(np.round(portions[1] * buffer_size, 0))
		max_len_others = int(np.round(portions[2] * buffer_size, 0))
		self.rpl_pos = ReplayBuffer(max_len_pos, view_size, action_hist.get_size())
		self.rpl_path_to_pos = ReplayBuffer(max_len_path_to_pos, view_size, action_hist.get_size())
		self.rpl_others = ReplayBuffer(max_len_others, view_size, action_hist.get_size())
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

		len_all_rpl = self.rpl_pos.len + self.rpl_path_to_pos.len + self.rpl_others.len

		if len_all_rpl >= self.start_learn and self.learn_steps % self.learn_interval == 0:
			minibatch = self._draw_minibatch(self.batch_size)
			self.model.update_qs(*minibatch)
			self.learn_steps = 1
		else:
			self.learn_steps += 1

	def _draw_minibatch(self, batch_size):
		len_pos = int(np.round(self.portions[0] * batch_size, 0))
		len_path_to_pos = int(np.round(self.portions[1] * batch_size, 0))
		len_others = int(np.round(self.portions[2] * batch_size, 0))
		mbs = []
		for len_batch, rpl_list in zip((len_pos, len_path_to_pos, len_others), (self.rpl_pos, self.rpl_path_to_pos, self.rpl_others)):
			mb = self._draw_from_one_list(rpl_list, len_batch)
			if mb is not None:
				mbs.append(mb)
		if len(mbs) == 1:
			return mbs[0]
		else:
			return tuple( [ np.vstack((mbs[0][x], mbs[1][x], mbs[2][x])) for x in range(len(mbs[0])) ] )

	def _draw_from_one_list(self, rpl_list, len_batch):
		curr_len = rpl_list.len
		if curr_len == 0:
			mb = None
		elif curr_len < len_batch:
			mb = rpl_list.draw_batch(curr_len)
		else:
			mb = rpl_list.draw_batch(len_batch)
		return mb


	def new_episode(self):
		final_reward = self.last_exps[-1][-2]
		l = len(self.last_exps)
		if final_reward == 1:
			for i in range(l):
				exp = self.last_exps[i]
				if i == l-1:
					self.rpl_pos.add(*exp)
				else:
					self.rpl_path_to_pos.add(*exp)
		else:
			for i in range(l):
				exp = self.last_exps[i]
				self.rpl_others.add(*exp)
		self.action_hist.new_episode()
