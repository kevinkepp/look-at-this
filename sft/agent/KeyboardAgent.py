from sft.agent.RobotAgent import RobotAgent
from sft.Actions import Actions
import matplotlib.pyplot as plt

import sys


class KeyboardAgent(RobotAgent):

	def __init__(self):
		plt.ion()
		plt.figure()

	@staticmethod
	def key_to_action(k):
		if k == 'w':
			return Actions.UP
		elif k == 's':
			return Actions.DOWN
		elif k == 'd':
			return Actions.RIGHT
		elif k == 'a':
			return Actions.LEFT
		elif k == 'q':
			sys.exit(0)
		else:
			return None

	def choose_action(self, curr_state, eps):
		# show view
		plt.imshow(curr_state.view)
		plt.draw()
		print("Action history:\n{0}".format(curr_state.actions))
		key = None
		while key is None:
			key = raw_input("Choose action - w:up, s:down, d:right, a:left, q:quit - ")
			if key is None:
				print("Invalid key - choose again")
		return self.key_to_action(key)

	def incorporate_reward(self, old_state, action, new_state, value):
		print("Reward: {0}".format(value))
