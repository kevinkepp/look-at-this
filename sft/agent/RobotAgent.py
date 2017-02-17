from abc import ABCMeta, abstractmethod


# abstract class for robot agent
class RobotAgent(object):
	__metaclass__ = ABCMeta

	# choose action based on current state
	# epsilon determines how random the action is chosen
	# return None means abort
	@abstractmethod
	def choose_action(self, curr_state, eps):
		pass

	# receive reward for a action that moved agent from old to new state
	# if new_state is terminal None will be given as value
	@abstractmethod
	def incorporate_reward(self, old_state, action, new_state, value):
		pass

	# signals the agent, that one training episode is over and the next starts
	@abstractmethod
	def new_episode(self):
		pass
