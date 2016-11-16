from abc import ABCMeta, abstractmethod


# abstract class for robot agent
class RobotAgent():
	__metaclass__ = ABCMeta

	# choose action based on current state
	# return None means abort
	@abstractmethod
	def choose_action(self, curr_state):
		pass

	# receive reward for a action that moved agent from old to new state
	@abstractmethod
	def incorporate_reward(self, old_state, action, new_state, value):
		pass

	@abstractmethod
	# signals start of epoch n
	def new_epoch(self, n):
		pass
