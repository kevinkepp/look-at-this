from readchar import readkey

from sft.agent.QAgent import QAgent
from sft.sim.OldSimulator import Actions


class KeyboardQAgent(QAgent):
	# alpha: learning rate
	# gamma: discount factor
	# q_init: initializer function for q values
	def __init__(self, actions, alpha, gamma, q_init):
		super(KeyboardQAgent, self).__init__(actions, alpha, gamma, q_init)

	@staticmethod
	def key_to_action(k):
		if k == "w":
			return Actions.UP
		elif k == "s":
			return Actions.DOWN
		elif k == "d":
			return Actions.RIGHT
		elif k == "a":
			return Actions.LEFT
		else:
			return None

	# overwrite
	def choose_action(self, curr_state):
		print(curr_state)
		print("Press arrow key")
		key = readkey()
		return self.key_to_action(key)

	def incorporate_reward(self, old_state, action, new_state, value):
		super(KeyboardQAgent, self).incorporate_reward(old_state, action, new_state, value)
