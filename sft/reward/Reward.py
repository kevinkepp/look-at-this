from abc import abstractmethod, ABCMeta


class Reward(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def get_reward(self, old_state, new_state, at_target, oob, steps_exceeded):
		pass
