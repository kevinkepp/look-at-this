import random

from lat.RobotAgent import RobotAgent


class RandomAgent(RobotAgent):
	def __init__(self, actions):
		self._actions = actions

	def choose_action(self, curr_state):
		return random.choice(self._actions)

	def incorporate_reward(self, old_state, action, new_state, value):
		pass

	def new_epoch(self):
		pass
