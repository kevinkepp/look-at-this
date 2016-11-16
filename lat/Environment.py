from abc import abstractmethod, ABCMeta


class Environment():
	__metaclass__ = ABCMeta

	@abstractmethod
	def run(self):
		pass

	@abstractmethod
	def get_current_state(self):
		pass

	@abstractmethod
	def _execute_action(self, action):
		pass
