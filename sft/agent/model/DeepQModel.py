from abc import ABCMeta, abstractmethod


class DeepQModel(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def predict_qs(self, state):
		pass

	@abstractmethod
	def update_qs(self, state, target):
		pass

	@abstractmethod
	def load_from_file(self, filepath):
		pass

	@abstractmethod
	def save_to_file(self, filepath):
		pass