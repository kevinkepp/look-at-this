from sft.reward.Reward import Reward
from sft import Rectangle
from sft import Size
from sft.sim.Simulator import Simulator
import numpy as np


class ContinuousReward(Reward):
	TARGET_VALUE = Simulator.TARGET_VALUE

	def __init__(self, reward_not_target, reward_target):
		self.r_no = reward_not_target
		self.r_yes = reward_target

	def is_at_target(self, view):
		middle_point = Rectangle(None, Size(view.shape)).get_middle()
		middle_value = middle_point.at_matrix(view)
		return middle_value == self.TARGET_VALUE

	def is_on_path(self, view):
		if np.sum(view)>0:
			return 0.01
		else:
			return self.r_no

	def get_reward(self, view, view2):
		if view2 is not None:
			if self.is_at_target(view2):
				return self.r_yes
			else:
				return np.sum(view2) * 0.003
		else:
			return self.r_no
		# check if the state represents the agent being on the target
		# if view2 is not None and self.is_at_target(view2):
		# 	return self.r_yes
		# else:
		# 	return self.is_on_path(view2)
