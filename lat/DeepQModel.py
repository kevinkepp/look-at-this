from abc import ABCMeta, abstractmethod


class DeepQModel(metaclass=ABCMeta):
	@abstractmethod
	def predict_qs(self, state):
		pass

	@abstractmethod
	def update_qs(self, state, target):
		pass
