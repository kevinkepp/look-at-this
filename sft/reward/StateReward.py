from sft.reward.Reward import Reward
import numpy as np


class StateReward(Reward):
	def __init__(self, reward_target, reward_oob, reward_steps_exceeded, reward_else, sum_view_flag, sum_view_factor, reward_on_path):
		self.reward_target = reward_target
		self.reward_oob = reward_oob
		self.reward_steps_exceeded = reward_steps_exceeded
		self.reward_else = reward_else
		self.sum_view_flag = sum_view_flag
		self.reward_on_path = reward_on_path
		self.sum_view_factor = sum_view_factor

	def get_reward(self, view, view2, at_target, oob, steps_exceeded):
		# check if the state represents the agent being on the target
		if at_target:
			return self.reward_target
		elif oob:
			return self.reward_oob
		elif steps_exceeded:
			return self.reward_steps_exceeded
		else:
			if not self.sum_view_flag and self.reward_on_path != self.reward_else and np.sum(view2) > 0:
				# on path
				return self.reward_on_path
			elif self.sum_view_flag:
				return np.sum(view2) * self.sum_view_factor
			else:
				return self.reward_else
