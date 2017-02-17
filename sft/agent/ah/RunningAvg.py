import theano

from sft import Size
from sft.Actions import Actions
from sft.agent.ah.ActionHistory import ActionHistory

import numpy as np


class RunningAvg(ActionHistory):
	def __init__(self, logger, n, factor):
		self.logger = logger
		self.n = n
		self.factor = factor
		self.actions = np.zeros([self.n, self.ACTION_WIDTH], dtype=theano.config.floatX)

	def get_size(self):
		return Size(self.n, self.ACTION_WIDTH)

	def get_history(self, all_actions):
		return self.actions

	def new_action(self, a):
		for i in range(1, self.n)[::-1]:
			self.actions[i] = (1 - self.factor) * self.actions[i] + self.factor * self.actions[i - 1]
		self.actions[0, :] = Actions.get_one_hot(a)
		# self.logger.log_parameter("actions", str(self.actions))

	def new_episode(self):
		self.actions = np.zeros([self.n, self.ACTION_WIDTH], dtype=theano.config.floatX)
