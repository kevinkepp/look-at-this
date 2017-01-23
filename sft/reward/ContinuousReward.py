from sft.reward.Reward import Reward
from sft import Rectangle
from sft import Size
from sft.sim.Simulator import Simulator
import numpy as np


class SumViewReward(Reward):
	TARGET_VALUE = Simulator.TARGET_VALUE

	def __init__(self, reward_not_target, reward_target, factor=0.003):
		self.r_no = reward_not_target
		self.r_yes = reward_target
		self.factor = factor

	def is_at_target(self, view):
		middle_point = Rectangle(None, Size(view.shape)).get_middle()
		middle_value = middle_point.at_matrix(view)
		return middle_value == self.TARGET_VALUE

	def get_reward(self, view, view2):
		if view2 is not None:
			if self.is_at_target(view2):
				return self.r_yes
			else:
				return np.sum(view2) * self.factor
		else:
			return self.r_no


class OnPathReward(Reward):
	TARGET_VALUE = Simulator.TARGET_VALUE

	def __init__(self, reward_not_target, reward_target, reward_on_path=0.01):
		self.r_no = reward_not_target
		self.r_yes = reward_target
		self.reward_on_path = reward_on_path

	def is_at_target(self, view):
		middle_point = Rectangle(None, Size(view.shape)).get_middle()
		middle_value = middle_point.at_matrix(view)
		return middle_value == self.TARGET_VALUE

	def is_on_path(self, view):
		if np.sum(view)>0:
			return self.reward_on_path
		else:
			return self.r_no

	def get_reward(self, view, view2):
		# check if the state represents the agent being on the target
		if view2 is not None:
			if self.is_at_target(view2):
				return self.r_yes
			else:
				return self.is_on_path(view2)
		else:
			return self.r_no
