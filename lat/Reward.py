from abc import abstractmethod, ABCMeta


class Reward(metaclass=ABCMeta):

	def get_reward(self, old_state, new_state):
		pass