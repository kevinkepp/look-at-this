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

	def get_size(self):
		return Size(self.n, self.ACTION_WIDTH)

	def get_history(self, all_actions):
		actions = np.zeros([self.n, self.ACTION_WIDTH], dtype=theano.config.floatX)
		# take last n actions, this will be smaller or empty if there are not enough actions
		for a in all_actions:
			actions[0, :] = Actions.get_one_hot(a)
			for i in range(1, self.n):
				actions[i] = self.normalize(actions[i] + self.factor * actions[i-1])
		self.logger.log_parameter("actions", str(actions))
		return actions

	def normalize(self, vec):
		return vec / float(np.max(vec))

