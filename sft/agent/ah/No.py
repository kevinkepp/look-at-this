import theano

from sft import Size
from sft.Actions import Actions
from sft.agent.ah.ActionHistory import ActionHistory

import numpy as np


class No(ActionHistory):
	def get_size(self):
		return Size(0, self.ACTION_WIDTH)

	def get_history(self, all_actions):
		return np.zeros([0, self.ACTION_WIDTH], dtype=theano.config.floatX)
