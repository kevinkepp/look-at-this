from __future__ import division

import random

import numpy as np
import theano

from sft.agent.RobotAgent import RobotAgent


class ReplayBuffer(object):
	def __init__(self, size, view_size, action_hist_size):
		self.size = size
		self.len = 0
		self.top = 0
		self.v = np.zeros((size, 1, view_size.w, view_size.h), dtype=theano.config.floatX)
		self.ah = np.zeros((size, 1, action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX)
		self.a = np.zeros((size, 1), dtype=np.int32)
		self.v2 = np.zeros((size, 1, view_size.w, view_size.h), dtype=theano.config.floatX)
		self.ah2 = np.zeros((size, 1, action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX)
		self.r = np.zeros((size, 1), dtype=theano.config.floatX)
		self.t = np.zeros((size, 1), dtype=np.int32)

	def add(self, v, ah, a, v2, ah2, r, t):
		# write to current index
		self.v[self.top] = v
		self.ah[self.top] = ah
		self.a[self.top] = a
		self.v2[self.top] = v2
		self.ah2[self.top] = ah2
		self.r[self.top] = r
		self.t[self.top] = t
		self.top += 1
		if self.len < self.size:
			self.len += 1
		if self.top == self.size:
			self.top = 0

	def draw_batch(self, batch_size):
		indices = np.random.randint(0, self.len, batch_size)
		return self.v[indices], self.ah[indices], self.a[indices], self.v2[indices], self.ah2[indices], \
			   self.r[indices], self.t[indices]


class DeepQAgentGpu(RobotAgent):
	# actions: possible actions
	# epsilon: epsilon-greedy strategy
	# epsilon: discount function for epsilon
	# batch size: size of minibatches for experience replay (see doi:10.1038/nature14236)
	# buffer size: size of experience pool from which minibatches are randomly sampled
	# start_learn: after how many experiences (buffer size) we start learning based on experiences
	# learn_steps: every learn_steps steps the model gets updated
	def __init__(self, logger, actions, batch_size, buffer_size, start_learn, learn_interval, view_size,
				 action_hist, model):
		self.logger = logger
		self.actions = actions
		self.batch_size = batch_size
		self.buffer_size = max(buffer_size, batch_size)  # buffer has to be at least batch_size
		self.start_learn = start_learn
		self.learn_interval = learn_interval if learn_interval > 0 else 1
		self.view_size = view_size
		self.action_hist = action_hist
		self.model = model
		self.replay_buffer = ReplayBuffer(buffer_size, view_size, action_hist.get_size())
		self.learn_steps = 0

	def choose_action(self, curr_state, eps):
		actions = self.action_hist.get_history(curr_state.actions)
		qs = self.model.predict_qs(curr_state.view, actions)
		self.logger.log_parameter("q", qs)
		# store qs for current state because usually we can use them in subsequent call to incorporate_reward
		if np.random.random() < eps:
			ai = np.random.randint(0, len(self.actions))
		else:
			ai = np.argmax(qs)
		action = self.actions[ai]
		self.action_hist.new_action(action)
		return action

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
		self.replay_buffer.add(*exp_new)
		if self.replay_buffer.len >= self.start_learn and self.learn_steps % self.learn_interval == 0:
			minibatch = self.replay_buffer.draw_batch(self.batch_size)
			self.model.update_qs(*minibatch)
			self.learn_steps = 1
		else:
			self.learn_steps += 1

	def new_episode(self):
		self.action_hist.new_episode()
