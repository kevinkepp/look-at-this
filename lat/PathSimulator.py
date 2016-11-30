import networkx as nx
import numpy as np
import cv2

from lat.Simulator import ImageSimulator, SimpleMatrixSimulator


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

	def tuple(self):
		return self.w, self.h


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


class PathSimulator(SimpleMatrixSimulator):
	WORLD_SIZE_MIN_W = 20
	WORLD_SIZE_MIN_H = 20

	PATH_LENGTH_MIN = 5
	PATH_LENGTH_MAX = 20
	PATH_LENGTH_MEAN = 12
	PATH_LENGTH_STD = 5

	def __init__(self, agent, reward, grid_n, grid_m=1, orientation=0, max_steps=1000,
											visualizer=None, bounded=True, world_size_factor=-1):
		super(PathSimulator, self).__init__(agent, reward, grid_n, grid_m, orientation, max_steps, visualizer, bounded)
		if world_size_factor > 1:
			self.world_factor = world_size_factor
		self.world_size = Size(grid_n * self.world_factor, grid_m * self.world_factor)
		if self.world_size.w < self.WORLD_SIZE_MIN_W or self.world_size.h < self.WORLD_SIZE_MIN_H:
			raise ValueError(
				"World size too small. Minimum is (%d,%d)" % (self.WORLD_SIZE_MIN_W, self.WORLD_SIZE_MIN_H))
		self.view_size = Size(grid_n, grid_m)
		if self.view_size.w < 1 or self.view_size.h < 1:
			raise ValueError("View size too small. Minimum is (1,1)")
		if self.view_size.w > self.world_size.w * 1.5 or self.view_size.h > self.world_size.h * 1.5:
			raise ValueError("View size too large. Maximum is 1.5 times world size.")
		self.bbox = self.get_bbox(self.world_size, self.view_size)

	def _initialize_world(self, path_length=-1):
		# new world state
		ws = np.full(self.world_size.tuple(), 0., dtype=np.float)
		self.graph = nx.Graph()
		# only add one path for now
		self.generate_path(length=path_length, path_id=0)
		self.render_paths(ws)
		# add target to random vertex
		node_target = self.generate_target()
		self.render_target(ws, node_target)
		# normalize image to [0, 1]
		ws = np.array(ws, np.float32)
		ws_min = np.min(ws)
		ws_max = np.max(ws)
		ws_diff = ws_max - ws_min
		if ws_diff != 0:
			ws -= ws_min
			ws /= ws_diff
		self.target = 1.
		self.world_state = ws

	def _get_init_state(self):
		self.state = None
		# we need to generate a new world for every run
		self._initialize_world()
		# find an initial state where a road is visible (sum of all visible pixel is not 0)
		while self.state is None or self._is_oob():
			view_pos = self.sample_view_position()
			self.i_world, self.j_world = view_pos.tuple()
			self.first_i, self.first_j = view_pos.tuple()
			self.state = self.world_state[view_pos.x:view_pos.x + self.view_size.w, view_pos.y:view_pos.y + self.view_size.h]
		# DEBUG, draw current view in world state
		# view = self.world_state.copy() * 255
		# cv2.rectangle(view, (view_pos.y, view_pos.x), (view_pos.y + self.view_size.h, view_pos.x + self.view_size.w),
		#			  (255, 255, 255), 1)
		# cv2.imwrite("tmp/view_curr.png", view)
		return self.state

	def sample_view_position(self):
		# sample view position
		i = self.sample_uniform(0, self.world_size.w - self.view_size.w)
		j = self.sample_uniform(0, self.world_size.h - self.view_size.h)
		return Point(i, j)

	def render_paths(self, img):
		for u, v in self.graph.edges():
			e = self.graph[u][v]
			color = 150
			thickness = 2
			# TODO dotted line from http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
			cv2.line(img, u.loc.tuple(), v.loc.tuple(), color=color, thickness=thickness)

	def get_bbox(self, world_size, view_size):
		border = Point(view_size.w, view_size.h)
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
			# sample node and only accept it if new edge does not intersect existing edges, maximally sample 20 times
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
			else:  # when loop ended normally we stop building path because we can't find node with non-intersecting edge anymore
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
	def sample_step_from(self, bbox, prev_node, step_size_min=-1):
		direction = self.sample_uniform(4)
		step = Point(0, 0)
		if step_size_min == -1:
			step_size_min = max(min(self.view_size.tuple()) / 2., 2)
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
		step_size_diff = step_size_max - step_size_min
		# check if step not possible
		if step_size_diff <= 1:
			return None
		step_size_mean = step_size_diff / 2 + step_size_min
		step_size_std = max(step_size_diff / 5., 1)
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

	def generate_target(self):
		nodes = self.graph.nodes()
		node = np.random.choice(nodes)
		self.graph[node]['target'] = 1
		return node

	def render_target(self, img, node):
		radius = 4
		color = 255
		cv2.circle(img, node.loc.tuple(), radius, color, -1)

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


class PathSimulatorSimple(PathSimulator):

	def _initialize_world(self, path_length=-1):
		# always generate straight line path with length
		length = 1
		super(PathSimulatorSimple, self)._initialize_world(length)

	def sample_step_from(self, bbox, prev_node, step_size_min=-1):
		# step size is minimum two views
		step_size_min = min(self.view_size.tuple()) * 2
		return super(PathSimulatorSimple, self).sample_step_from(bbox, prev_node, step_size_min)

class PathSimSimpleExpansiveSampler(PathSimulatorSimple):
	""" simple path simulation -> just a line with target on it + expansive sampling """
	# IMPORTANT IF NAME IS CHANGED ALSO LOOK INTO 'RUN' OF EVALUATOR TO CHANGE NAME

	epochs = None  # contains the overall max of epochs
	curr_epoch = 1  # contains current number of epochs

	pct_epochs_until_max = 0.7  # after what percentage of epochs to reach max sampling distance
	pct_min_dist_in_grid_size = 0.4  # at which distance to start sampling (in percent of grid-size)

	def _get_init_state(self):
		""" use special sampling  """
		self.state = None
		# we need to generate a new world for every run
		self._initialize_world()
		self._add_border_of_zeros()  # TODO: talk about that, padding 1.5 grid size at top and bottom
		# here special expansive sampling happens
		# get parameters
		i, j = self._get_target_loc()
		N, M = self.world_state.shape
		n, m = self.grid_dims
		# get limits
		i_lim, j_lim = self._get_limits(i, j, N, M, n, m)
		# get start position of upper left corner of view
		i_0 = np.random.randint(i_lim[0], i_lim[1]+1)
		j_0 = np.random.randint(j_lim[0], j_lim[1] + 1)
		# set state, start-state and extract the view
		self.i_world, self.j_world = i_0, j_0
		self.first_i, self.first_j = i_0, j_0
		self.state = self._extract_state_from_world(i_0, j_0)
		self.curr_epoch += 1
		return self.state

	def _get_limits(self, i_goal, j_goal, n_world, m_world, n_grid, m_grid):
		i_lim = self._get_limit(i_goal, n_grid, n_world)
		j_lim = self._get_limit(j_goal, m_grid, m_world)
		return i_lim, j_lim

	def _get_limit(self, i, n, N):
		e = self.curr_epoch
		e_max_reached = self.epochs * self.pct_epochs_until_max
		if e >= e_max_reached:
			i_lim = (n, N-2*n)
		else:
			min_dist = n * self.pct_min_dist_in_grid_size
			max_dist = N
			dist = int((max_dist - min_dist - i - n)*e/e_max_reached)
			i_low = i - min_dist - dist - int(n/2)
			i_high = i + min_dist + dist - int(n/2)
			i_lim = (i_low, i_high)
			i_lim = self._correct_to_within_world_state(i_lim, n, N)
		return i_lim

	def _correct_to_within_world_state(self, i_lim, n, N):
		i_low = i_lim[0]
		i_high = i_lim[1]
		if i_low - n < 0:
			i_low = n
		if i_high > N - 2*n:
			i_high = N - 2*n
		return i_low, i_high

	def _get_target_loc(self):
		# TODO: there must be a better way to extract the goal location (middle of circle)
		ij_target = np.where(self.world_state == 1)
		ij_target = np.mean(ij_target, axis=1).astype(int)
		assert ij_target.size == 2, "target location in expansive sampler does not contain 2 indices"
		return ij_target

	# somehow needs to know EPOCHS and current epoch

	def restartExpansiveSampling(self, epochs):
		self.epochs = epochs
		self.curr_epoch = 1

	# is out of bounds (if at edge of world_state this is already oob)
	def _is_oob(self):
		(N, M) = self.world_state.shape
		(n, m) = self.grid_dims
		if self.i_world <= 0 or self.j_world <= 0 or self.i_world + n >= N or self.j_world + m >= M:
			return True
		else:
			return False

	def _add_border_of_zeros(self):
		pad_grid_factor = 1.5
		pad_n, pad_m = (np.array(self.grid_dims) * pad_grid_factor).astype(int)
		self.world_state = np.lib.pad(self.world_state, ((pad_n, pad_n), (pad_m, pad_m)), 'constant', constant_values=0)

	# TODO: workaround to expand bbox, padding zeros later (see _add_border_of_zeros above)
	def get_bbox(self, world_size, view_size):
		border = Point(view_size.w, view_size.h)
		# size = Size(world_size.w - 2 * border.x, world_size.h - 2 * border.y)
		size = Size(world_size.w - border.x, world_size.h - border.x)
		return BoundingBox(border, size)

	# # TODO: workaround to increase the length of paths (it would be good to have some longer paths, that extend through the entire image)
	# def sample_normal(self, mean, std, low=float("-inf"), high=float("inf")):
	# 	while True:
	# 		idx = int(np.random.normal(mean*1.5, std, 1))
	# 		if low <= idx <= high*1.5:
	# 			return idx
	# 	return float("nan")
