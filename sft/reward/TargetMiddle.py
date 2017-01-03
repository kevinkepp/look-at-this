from sft.reward.Reward import Reward
from sft import Rectangle
from sft import Size
from sft.sim.Simulator import Simulator


class TargetMiddle(Reward):
	TARGET_VALUE = Simulator.TARGET_VALUE

	def __init__(self, reward_not_target, reward_target):
		self.r_no = reward_not_target
		self.r_yes = reward_target

	def is_at_target(self, view):
		middle_point = Rectangle(None, Size(view.shape)).get_middle()
		middle_value = middle_point.at_matrix(view)
		return middle_value == self.TARGET_VALUE

	def get_reward(self, view, view2):
		# check if the state represents the agent being on the target
		if view2 is not None and self.is_at_target(view2):
			return self.r_yes
		else:
			return self.r_no
