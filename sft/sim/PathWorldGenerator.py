import cv2
import networkx as nx
import numpy as np

from sft import Rectangle, normalize, get_bbox, sample_int_normal_bounded
from sft.Scenario import Scenario
from sft.sim.PathGenerator import PathGenerator
from sft.sim.ScenarioGenerator import ScenarioGenerator
from sft.sim.Simulator import Simulator


class PathWorldGenerator(ScenarioGenerator):
	PATH_LENGTH_MIN = 5
	PATH_LENGTH_MAX = 20
	PATH_THICKNESS = 1
	PATH_COLOR = 150

	TARGET_COLOR = int(Simulator.TARGET_VALUE * 255)
	TARGET_RADIUS = 1

	def __init__(self, logger, view_size, world_size, sampler, path_length_min=-1, path_length_max=-1,
				 path_step_length_min=-1, path_step_length_max=-1, path_in_init_view=False,
				 target_not_in_init_view=False):
		self.logger = logger
		self.view_size = view_size
		self.world_size = world_size
		self.bbox = get_bbox(world_size, view_size)
		self.sampler = sampler
		self.path_length_min = path_length_min
		self.path_length_max = path_length_max
		self.generator = PathGenerator(self.view_size, self.bbox, path_step_length_min, path_step_length_max)
		self.path_in_init_view = path_in_init_view
		self.target_not_in_init_view = target_not_in_init_view

	def get_next(self):
		world, target_pos = self.init_world()
		view_pos = self.init_pos(world, target_pos)
		return Scenario(world, view_pos)

	def init_world(self):
		world = np.full(self.world_size.tuple(), 0., dtype=np.float)
		graph = nx.Graph()
		self.init_path(world, graph)
		target_pos = self.init_target(world, graph)
		world = np.array(world, np.float32)
		# normalize world image to [0, 1]
		normalize(world)
		return world, target_pos

	def init_path(self, world, graph):
		# only add one path for now
		# sample length for path
		self.generator.generate_path(self.sample_path_length(), graph, path_id=0)
		self.render_paths(world, graph)

	def sample_path_length(self):
		l_min = self.path_length_min if self.path_length_min != -1 else self.PATH_LENGTH_MIN
		l_max = self.path_length_max if self.path_length_max != -1 else self.PATH_LENGTH_MAX
		l_max = max(l_min, l_max)
		return sample_int_normal_bounded(l_min, l_max)

	def render_paths(self, image, graph):
		for u, v in graph.edges():
			e = graph[u][v]
			# here we can get edge attributes from e
			# TODO dotted line from http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
			cv2.line(image, pt1=u.pos.tuple(), pt2=v.pos.tuple(), color=self.PATH_COLOR, thickness=self.PATH_THICKNESS)

	def init_target(self, world, graph):
		node_target = self.generate_target(graph)
		self.render_target(world, node_target)
		# return target position
		return node_target.pos

	def generate_target(self, graph):
		# pick one of the graph nodes as target
		nodes = graph.nodes()
		node = np.random.choice(nodes)
		graph[node]['target'] = 1
		return node

	def render_target(self, image, node):
		# thickness -1 means fill circle
		cv2.circle(image, center=node.pos.tuple(), radius=self.TARGET_RADIUS, color=self.TARGET_COLOR, thickness=-1)

	def init_pos(self, world, target_pos):
		pos = None
		view = None
		# when required find initial view where path is visible
		while view is None or not self.is_view_valid(view):
			pos = self.sampler.sample_init_pos(self.bbox, target_pos)
			view = self.get_view(world, pos)
		return pos

	def is_view_valid(self, view):
		path_valid = self.is_path_in_view(view) if self.path_in_init_view else True
		target_valid = not self.is_target_in_view(view) if self.target_not_in_init_view else True
		return path_valid and target_valid

	def get_view(self, world, pos):
		# find start of view because view_pos indicates center of view
		view_pos_start = pos - Rectangle(None, self.view_size).get_middle()
		view = Rectangle(view_pos_start, self.view_size).crop_matrix(world)
		return view

	def is_path_in_view(self, view):
		# for now just check if there are non-zero pixel
		return np.sum(view) > 0.

	def is_target_in_view(self, view):
		return len(np.where(view == 1.)[0])
