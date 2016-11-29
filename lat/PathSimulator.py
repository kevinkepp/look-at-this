import networkx as nx
import numpy as np
import cv2


class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __add__(self, other):
		if isinstance(other, Point):
			return Point(self.x + other.x, self.y + other.y)
		elif isinstance(other, (float, int, long)):
			return Point(self.x + other, self.y + other)
		else:
			return NotImplemented

	def __sub__(self, other):
		if isinstance(other, Point):
			return Point(self.x - other.x, self.y - other.y)
		elif isinstance(other, (float, int, long)):
			return Point(self.x - other, self.y - other)
		else:
			return NotImplemented

	def __mul__(self, other):
		if isinstance(other, Point):
			return Point(self.x * other.x, self.y * other.y)
		elif isinstance(other, (float, int, long)):
			return Point(self.x * other, self.y * other)
		else:
			return NotImplemented

	def __div__(self, other):
		if isinstance(other, Point):
			return Point(self.x / other.x, self.y / other.y)
		elif isinstance(other, (float, int, long)):
			return Point(self.x / other, self.y / other)
		else:
			return NotImplemented

	def tuple(self):
		return self.x, self.y

	def __str__(self):
		return "({0},{1})".format(self.x, self.y)

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		return NotImplemented

	def __ne__(self, other):
		if isinstance(other, self.__class__):
			return not self.__eq__(other)
		return NotImplemented

	def __hash__(self):
		return hash(tuple(sorted(self.__dict__.items())))


class Size:
	def __init__(self, w, h):
		self.w = w
		self.h = h


class BoundingBox:
	def __init__(self, point, size):
		self.point = point
		self.x = point.x
		self.y = point.y
		self.size = size
		self.w = size.w
		self.h = size.h

	def contains(self, point):
		return self.x <= point.x <= self.x + self.w and self.y <= point.y <= self.y + self.h


class PathNode:
	def __init__(self, id_, loc, path_id=-1):
		self.id = id_
		self.path_id = path_id
		self.loc = loc


class Simulator:
	WORLD_SIZE_MIN_W = 20
	WORLD_SIZE_MIN_H = 20

	PATH_LENGTH_MIN = 5
	PATH_LENGTH_MAX = 20
	PATH_LENGTH_MEAN = 12
	PATH_LENGTH_STD = 5

	def __init__(self, world_size, view_size):
		self.world_size = Size(world_size[0], world_size[1])
		if self.world_size.w < self.WORLD_SIZE_MIN_W or self.world_size.h < self.WORLD_SIZE_MIN_H:
			raise ValueError(
				"World size too small. Minimum is (%d,%d)" % (self.WORLD_SIZE_MIN_W, self.WORLD_SIZE_MIN_H))
		self.view_size = Size(view_size[0], view_size[1])
		if self.view_size.w < 1 or self.view_size.h < 1:
			raise ValueError("View size too small. Minimum is (1,1)")
		if self.view_size.w > self.world_size.w * 1.5 or self.view_size.h > self.world_size.h * 1.5:
			raise ValueError("View size too large. Maximum is 1.5 times world size.")
		self.img = np.full(world_size, 0, dtype=np.float)
		self.bbox = self.get_bbox(self.world_size, self.view_size)
		self.graph = nx.Graph()
		# only add one path for now
		self.generate_path(path_id=0)
		self.render_paths(self.img)
		cv2.imwrite("tmp/out.png", self.img)

	def render_paths(self, img):
		for u, v in self.graph.edges():
			e = self.graph[u][v]
			color = (200, 200, 200)
			thickness = 2
			# TODO dotted line from http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
			cv2.line(img, u.loc.tuple(), v.loc.tuple(), color=color, thickness=thickness)

	def get_bbox(self, world_size, view_size):
		border = Point(view_size.w / 2, view_size.h / 2)
		size = Size(world_size.w - 2 * border.x, world_size.h - 2 * border.y)
		return BoundingBox(border, size)

	def generate_path(self, nodes=None, length=-1, path_id=-1):
		if length == -1:
			# sample length of path
			length = self.sample_normal(self.PATH_LENGTH_MEAN, self.PATH_LENGTH_STD, self.PATH_LENGTH_MIN,
										self.PATH_LENGTH_MAX)
		if nodes is None or not isinstance(nodes, list):
			nodes = []
		if len(nodes) == 0:
			# sample starting node uniformly
			loc = self.sample_point(self.bbox)
			node_new = PathNode(0, loc, path_id)
		else:
			# sample next node based on previous one
			id_ = len(self.graph.nodes())
			# only accept node if new edge does not intersect existing edges, maximally sample 20 times
			for i in range(20):
				# try to sample step, maximally 10 times
				for j in range(10):
					loc = self.sample_step_from(self.bbox, nodes[-1])
					if loc is not None:
						break
				else:  # when loop ended normally we stop building path because we can't sample new steps anymore
					return
				node_new = PathNode(id_, loc, path_id)
				if not self.is_intersecting_any((nodes[-1], node_new), self.graph.edges()):
					break
			else:  # when loop ended normally we stop building path because we can't find non intersecting node anymore
				return
		# add new node to the graph
		self.graph.add_node(node_new)
		# add edge from previous node to new node
		if len(nodes) > 0:
			self.graph.add_edge(nodes[-1], node_new)
		# recurse if length not reached
		nodes.append(node_new)
		if length > 0:
			self.generate_path(nodes, length - 1, path_id)

	# samples a location for a new node based on a given previous node
	def sample_step_from(self, bbox, prev_node):
		step_size_min = 1
		direction = self.sample_uniform(4)
		step = Point(0, 0)
		if direction == 0:  # up
			step_size_max = prev_node.loc.y - bbox.y
			step.y = -1
		elif direction == 1:  # down
			step_size_max = bbox.y + bbox.h - prev_node.loc.y
			step.y = 1
		elif direction == 2:  # left
			step_size_max = prev_node.loc.x - bbox.x
			step.x = -1
		else:  # right
			step_size_max = bbox.x + bbox.w - prev_node.loc.x
			step.x = 1
		# check if step not possible
		if step_size_max <= 1:
			return None
		step_size_mean = np.mean([step_size_min, step_size_max])
		step_size_std = (step_size_max - step_size_min) / 5.
		step_size = self.sample_normal(step_size_mean, step_size_std, step_size_min, step_size_max)
		step *= step_size
		return prev_node.loc + step

	def is_intersecting_any(self, edge_new, edges):
		for e in edges:
			if self.is_intersecting(edge_new, e):
				return True
		return False

	def is_intersecting(self, e1, e2):
		v1, v2 = e1
		v3, v4 = e2
		p1 = v1.loc
		p2 = v2.loc
		p3 = v3.loc
		p4 = v4.loc
		# Before anything else check if lines have a mutual abcisses
		interval_1 = [min(p1.x, p2.x), max(p1.x, p2.x)]
		interval_2 = [min(p3.x, p4.x), max(p3.x, p4.x)]
		interval = [
			min(interval_1[1], interval_2[1]),
			max(interval_1[0], interval_2[0])
		]
		if interval_1[1] < interval_2[0]:
			return False  # no mutual abcisses
		# Try to compute interception
		def line(p1, p2):
			A = (p1.y - p2.y)
			B = (p2.x - p1.x)
			C = (p1.x * p2.y - p2.x * p1.y)
			return A, B, -C
		L1 = line(p1, p2)
		L2 = line(p3, p4)
		D = L1[0] * L2[1] - L1[1] * L2[0]
		Dx = L1[2] * L2[1] - L1[1] * L2[2]
		Dy = L1[0] * L2[2] - L1[2] * L2[0]
		if D != 0:
			x = Dx / D
			y = Dy / D
			p = Point(x, y)
			if p.x < interval[1] or p.x > interval[0]:
				return False  # out of bounds
			else:
				# it's an intersection if new edge does not originate from previous one
				return not (p == p1 == p3 or p == p1 == p4)
		else:
			return False  # not intersecting

	def sample_point(self, bbox):
		x = self.sample_uniform(bbox.x, bbox.x + bbox.w)
		y = self.sample_uniform(bbox.y, bbox.y + bbox.h)
		return Point(x, y)

	def sample_uniform(self, low, high=None):
		return np.random.randint(low, high)

	def sample_normal(self, mean, std, low=float("-inf"), high=float("inf")):
		while True:
			idx = int(np.random.normal(mean, std, 1))
			if low <= idx <= high:
				return idx
		return float("nan")

	def is_oob(self, bbox, point):
		return not bbox.contains(point)
