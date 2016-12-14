from abc import abstractmethod, ABCMeta


class Environment(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def run(self, agent):
		pass

	@abstractmethod
	def get_current_state(self):
		pass

	@abstractmethod
	def execute_action(self, action):
		pass
