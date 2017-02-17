class State:
	"""view and actions are matrices"""

	def __init__(self, view, actions):
		self.view = view
		self.actions = actions

	def to_list(self):
		return [self.view, self.actions]
