from abc import ABCMeta, abstractmethod


# abstract class for robot agent
class RobotAgent(metaclass=ABCMeta):
	# choose action based on current state
	# return None means abort
	@abstractmethod
	def choose_action(self, curr_state):
		pass

	# receive reward for a action that moved agent from old to new state
	# new_state is None if terminated
	@abstractmethod
	def incorporate_reward(self, old_state, action, new_state, value):
		pass

	@abstractmethod
	def new_epoch(self):
		pass
