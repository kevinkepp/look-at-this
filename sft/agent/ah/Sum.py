import theano

from sft import Size
from sft.Actions import Actions
from sft.agent.ah.ActionHistory import ActionHistory

import numpy as np


class Sum(ActionHistory):
	def __init__(self, logger):
		self.logger = logger
		self.actions = np.zeros([1, self.ACTION_WIDTH], dtype=theano.config.floatX)

	def get_size(self):
		return Size(1, self.ACTION_WIDTH)

	def get_history(self, all_actions):
		return self.actions

	def new_action(self, a):
		self.actions += Actions.get_one_hot(a)
		# normalize vector to [0, 1]
		self.actions /= float(np.max(self.actions)) if np.max(self.actions) != 0 else 1

	def new_episode(self):
		self.actions = np.zeros([1, self.ACTION_WIDTH], dtype=theano.config.floatX)

