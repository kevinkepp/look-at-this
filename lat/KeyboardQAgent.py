from lat.QAgent import QLearningAgent
from lat.RobotAgent import Actions
import readchar


class KeyboardQAgent(QLearningAgent):
	# alpha: learning rate
	# gamma: discount factor
	# q_init: initializer function for q values
	def __init__(self, alpha, gamma, q_init):
		super(KeyboardQAgent, self).__init__(alpha, gamma, q_init)

	@staticmethod
	def key_to_action(k):
		if k == '\x1b[A':
			return Actions.up
		elif k == '\x1b[B':
			return Actions.down
		elif k == '\x1b[C':
			return Actions.right
		elif k == '\x1b[D':
			return Actions.left
		else:
			return None

	# overwrite
	def choose_action(self, curr_state):
		print(curr_state)
		key = readchar.readkey()
		return self.key_to_action(key)

	def incorporate_reward(self, old_state, action, new_state, value):
		super(KeyboardQAgent, self).incorporate_reward(old_state, action, new_state, value)
