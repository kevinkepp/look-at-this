from abc import abstractmethod, ABCMeta


class Environment(metaclass=ABCMeta):
	@abstractmethod
	def run(self):
		pass

	@abstractmethod
	def get_current_state(self):
		pass

	@abstractmethod
	def execute_action(self, action):
		pass
