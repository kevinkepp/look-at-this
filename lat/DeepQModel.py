from abc import ABCMeta, abstractmethod


class DeepQModel():
	__metaclass__ = ABCMeta

	@abstractmethod
	def predict_qs(self, state):
		pass

	@abstractmethod
	def update_qs(self, state, target):
		pass

	@abstractmethod
	def load_weights(self, filepath):
		pass

	@abstractmethod
	def save_weights(self, filepath):
		pass