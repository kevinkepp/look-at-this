class Actions(object):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3

	name_dict = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}

	all = [UP, DOWN, LEFT, RIGHT]

	@staticmethod
	def get_opposite(action):
		if action == Actions.UP:
			return Actions.DOWN
		elif action == Actions.DOWN:
			return Actions.UP
		elif action == Actions.LEFT:
			return Actions.RIGHT
		elif action == Actions.RIGHT:
			return Actions.LEFT

	@staticmethod
	def get_one_hot(action):
		if action == Actions.UP:
			return [0, 1, 0, 0]
		elif action == Actions.DOWN:
			return [0, 0, 0, 1]
		elif action == Actions.LEFT:
			return [1, 0, 0, 0]
		elif action == Actions.RIGHT:
			return [0, 0, 1, 0]
