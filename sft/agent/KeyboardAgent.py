from sft.agent.RobotAgent import RobotAgent
from sft.Actions import Actions


class KeyboardAgent(RobotAgent):

	def __init__(self):
		# default the agent starts in trainingmode
		self.trainingmode = True

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
		else:
			return None

	# overwrite
	def choose_action(self, curr_state, eps):
		#print(curr_state)
		key = raw_input("w:up, s:down, d:right, a:left, input your choice and press enter: ")
		return self.key_to_action(key)

	def incorporate_reward(self, old_state, action, new_state, value):
		pass

	# return if the agent currently is in training mode or not
	def is_in_training_mode(self):
		return self.trainingmode

	# change the training mode
	def set_training_mode(self, in_training):
		self.trainingmode = in_training

	def new_epoch(self, n):
		pass
