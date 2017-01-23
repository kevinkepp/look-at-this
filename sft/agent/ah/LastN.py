import theano

from sft import Size
from sft.Actions import Actions
from sft.agent.ah.ActionHistory import ActionHistory

import numpy as np


class LastN(ActionHistory):
	def __init__(self, logger, n):
		self.logger = logger
		self.n = n

	def get_size(self):
		return Size(self.n, self.ACTION_WIDTH)

	def get_history(self, all_actions):
		actions = np.zeros([self.n, self.ACTION_WIDTH], dtype=theano.config.floatX)
		# take last n actions, this will be smaller or empty if there are not enough actions
		last_n_actions = list(reversed(all_actions[-self.n:])) if self.n > 0 else []
		for i in range(len(last_n_actions)):
			action = last_n_actions[i]
			actions[i] = Actions.get_one_hot(action)
		return actions

