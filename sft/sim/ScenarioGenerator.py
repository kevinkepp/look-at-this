from abc import ABCMeta, abstractmethod


class ScenarioGenerator(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def get_next(self):
		pass
