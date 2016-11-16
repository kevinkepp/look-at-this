from abc import abstractmethod, ABCMeta


class Reward():
	__metaclass__ = ABCMeta

	@abstractmethod
	def get_reward(self, old_state, new_state):
		pass
