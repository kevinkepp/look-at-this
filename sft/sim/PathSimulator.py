import os

import cv2
import networkx as nx
import numpy as np

from sft import Point, Size, Rectangle, normalize, submatrix
from sft.Actions import Actions
from sft.sim import sample_normal, sample_uniform, sample_point_within
from sft.sim.MatrixSimulator import MatrixSimulator
from sft.sim.PathGenerator import PathGenerator
from sft.sim.SimulatorNew import Simulator


class PathSimulator(MatrixSimulator):
	PATH_LENGTH_MIN = 5
	PATH_LENGTH_MAX = 20
	PATH_LENGTH_MEAN = 12
	PATH_LENGTH_STD = 5
	PATH_THICKNESS = 1
	PATH_COLOR = 150

	TARGET_COLOR = int(MatrixSimulator.TARGET_VALUE * 255)
	TARGET_RADIUS = 2

	def __init__(self, view_size, world_size, path_in_init_view=False):
		super(PathSimulator, self).__init__(view_size, world_size)
		self.generator = PathGenerator(view_size, self.bbox)
		self.graph = None
		self.path_in_init_view = path_in_init_view
		self.init_world()
		self.init_view()

	def init_world(self):
		# new world state
		self.world_image = np.full(self.world_size.tuple(), 0., dtype=np.float)
		self.graph = nx.Graph()
		self.init_path()
		self.init_target()
		self.world_image = np.array(self.world_image, np.float32)
		# normalize world image to [0, 1]
		normalize(self.world_image)

	def init_path(self):
		# only add one path for now
		# sample length for path
		path_length = sample_normal(self.PATH_LENGTH_MEAN, self.PATH_LENGTH_STD, self.PATH_LENGTH_MIN,
									self.PATH_LENGTH_MAX)
		self.generator.generate_path(path_length, self.graph, path_id=0)
		self.render_paths(self.world_image)

	def init_target(self):
		node_target = self.generate_target()
		self.render_target(self.world_image, node_target)

	def render_paths(self, img):
		for u, v in self.graph.edges():
			e = self.graph[u][v]
			# here we can get edge attributes from e
			# TODO dotted line from http://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
			cv2.line(img, pt1=u.pos.tuple(), pt2=v.pos.tuple(), color=self.PATH_COLOR, thickness=self.PATH_THICKNESS)

	def init_view(self):
		# when required find initial view where path is visible
		while self.view_image is None or (self.path_in_init_view and not self.is_path_in_view()):
			self.view_pos = sample_point_within(self.bbox)
			self.update_view_image()

	def update_view_image(self):
		# find start of view because view_pos indicates center of view
		view_start = self.view_pos - Point(self.view_size.w / 2, self.view_size.h / 2)
		self.view_image = submatrix(self.world_image, Rectangle(view_start, self.view_size))

	def is_path_in_view(self):
		# for now just check if there are non-zero pixel
		return np.sum(self.view_image) > 0.

	def generate_target(self):
		# pick one of the graph nodes as target
		nodes = self.graph.nodes()
		node = np.random.choice(nodes)
		self.graph[node]['target'] = 1
		return node

	def render_target(self, img, node):
		# thickness -1 means fill circle
		cv2.circle(img, center=node.pos.tuple(), radius=self.TARGET_RADIUS, color=self.TARGET_COLOR, thickness=-1)

	def reset(self):
		super(PathSimulator, self).reset()
		if self.graph is not None:
			self.graph.clear()
		self.graph = None
		self.init_world()
		self.init_view()
