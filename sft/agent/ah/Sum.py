import theano

from sft import Size
from sft.Actions import Actions
from sft.agent.ah.ActionHistory import ActionHistory

import numpy as np


class Sum(ActionHistory):
	def __init__(self, logger):
		self.logger = logger

	def get_size(self):
		return Size(1, self.ACTION_WIDTH)

	def get_history(self, all_actions):
		actions = np.zeros([1, self.ACTION_WIDTH], dtype=theano.config.floatX)
		# sum the one-hot representations
		for a in all_actions:
			actions += Actions.get_one_hot(a)
		# normalize vector to [0, 1]
		actions /= float(np.max(actions))
		return actions

