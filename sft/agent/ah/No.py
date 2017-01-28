import theano

from sft import Size
from sft.Actions import Actions
from sft.agent.ah.ActionHistory import ActionHistory

import numpy as np


class No(ActionHistory):

	def __init__(self):
		self.actions = np.zeros([0, self.ACTION_WIDTH], dtype=theano.config.floatX)

	def get_size(self):
		return Size(self.actions.shape)

	def get_history(self, all_actions):
		return self.actions

	def new_action(self, action):
		pass

	def new_episode(self):
		pass
