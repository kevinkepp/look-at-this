from abc import ABCMeta, abstractmethod

from sft.Actions import Actions


class ActionHistory(object):
	__metaclass__ = ABCMeta

	ACTION_WIDTH = len(Actions.all)

	@abstractmethod
	def get_size(self):
		pass

	@abstractmethod
	def get_history(self, all_actions):
		pass

	@abstractmethod
	def new_action(self, action):
		pass

	@abstractmethod
	def new_episode(self):
		pass
