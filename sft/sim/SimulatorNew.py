from abc import ABCMeta, abstractmethod

class Simulator(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def get_current_view(self):
		pass

	@abstractmethod
	def update_view(self, action):
		pass

	@abstractmethod
	def is_oob(self):
		pass

	@abstractmethod
	def is_at_target(self):
		pass

	@abstractmethod
	def reset(self):
		pass
