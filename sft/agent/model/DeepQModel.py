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
	def load(self, file_path):
		pass

	@abstractmethod
	def save(self, file_path):
		pass

	@abstractmethod
	def copy_from(self, file_path):
		pass
