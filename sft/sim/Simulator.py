from sft import Point, Size, Rectangle, submatrix, normalize, bbox
from sft.Actions import Actions


class Simulator(object):
	TARGET_VALUE = 1.

	def __init__(self, view_size):
		assert isinstance(view_size, Size)
		self.view_size = view_size
		self.world_size = None
		self.world_image = None
		self.view_pos = None  # center of view
		self.view_image = None
		self.bbox = None

	def initialize(self, world, pos):
		normalize(world)
		self.world_size = Size(world.shape)
		self.bbox = bbox(self.world_size, self.view_size)
		self.world_image = world
		self.view_pos = pos
		self.update_view_image()

	def update_view_image(self):
		if self.is_oob():
			self.view_image = None
		# find start of view because view_pos indicates center of view
		view_start = self.view_pos - Point(self.view_size.w / 2, self.view_size.h / 2)
		self.view_image = submatrix(self.world_image, Rectangle(view_start, self.view_size))

	def get_current_view(self):
		return self.view_image

	def get_current_pos(self):
		return self.view_pos

	def update_view(self, action):
		x = 0
		y = 0
		if action == Actions.UP:
			y = -1
		elif action == Actions.DOWN:
			y = 1
		elif action == Actions.LEFT:
			x = -1
		elif action == Actions.RIGHT:
			x = 1
		self.view_pos += Point(x, y)
		self.update_view_image()
		return self.get_current_view()

	def is_oob(self):
		return not self.bbox.contains(self.view_pos)

	def is_at_target(self):
		return self.world_image[self.view_pos.x, self.view_pos.y] == self.TARGET_VALUE
