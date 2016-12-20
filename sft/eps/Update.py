from abc import ABCMeta, abstractmethod


class Update(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def get_value(self, epoch):
		pass
