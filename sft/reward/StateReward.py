from sft.reward.Reward import Reward
from sft.sim.Simulator import Simulator


class TargetMiddle(Reward):
	TARGET_VALUE = Simulator.TARGET_VALUE

	def __init__(self, reward_target, reward_oob, reward_steps_exceeded, reward_else):
		self.reward_target = reward_target
		self.reward_oob = reward_oob
		self.reward_steps_exceeded = reward_steps_exceeded
		self.reward_else = reward_else

	def get_reward(self, view, view2, at_target, oob, steps_exceeded):
		# check if the state represents the agent being on the target
		if at_target:
			return self.reward_target
		elif oob:
			return self.reward_oob
		elif steps_exceeded:
			return self.reward_steps_exceeded
		else:
			return self.reward_else
