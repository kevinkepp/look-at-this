import numpy as np


class State:
	"""view and actions are matrices"""
	def __init__(self, view, actions):
		self.view = view
		self.actions = actions

	def flatten(self):
		"""reshapes state to row-vector (1,m) with m being number of features"""
		view = self.view.reshape((1, self.view.size))
		actions = self.actions.reshape((1, self.actions.size))
		return np.hstack((view, actions))

	@staticmethod
	def flatten_states(states):
		"""reshapes each state in the given list to row-vector and combine to matrix with shape (n,m) with
		n being number of states and m being number of features"""
		states = [s.flatten() for s in states]
		states = [s.reshape(s.size) for s in states]
		return np.matrix(states)

	def to_list(self):
		return [self.view, self.actions]
