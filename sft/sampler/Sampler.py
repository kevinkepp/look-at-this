from abc import ABCMeta, abstractmethod


class Sampler(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def sample_init_pos(self, bbox, target_pos):
		pass

	@abstractmethod
	def next_epoch(self):
		pass
