import cv2, os, random
import numpy as np
import theano

from sft import Point, Rectangle, get_path_bbox
from sft.Scenario import Scenario
from sft.sim.ScenarioGenerator import ScenarioGenerator


class PathWorldLoader(ScenarioGenerator):

	WOLRD_IMG_FORMAT = ".png"
	TARGET_VALUE = 1.

	def __init__(self, logger, world_path, view_size, world_size, sampler, path_in_init_view=False, target_not_in_init_view=False, path_init_file=None):
		self.logger = logger
		self.view_size = view_size
		self.world_size = world_size
		self.bbox = get_path_bbox(world_size, view_size)
		self.sampler = sampler
		self.path_in_init_view = path_in_init_view
		self.target_not_in_init_view = target_not_in_init_view
		self.worlds = {}  # used to store all loaded worlds
		self.path_init_file = path_init_file
		self._load_worlds(world_path)
		self.i_curr_world = 0

	def get_next(self, random_choice=True):
		# choose a world
		if len(self.worlds) == 1:
			world = self.worlds[0]
		elif random_choice:
			self.i_curr_world = random.choice(self.worlds.keys())
			world = self.worlds[self.i_curr_world]
		else:
			world = self.worlds[self.i_curr_world]
		# get init pos
		view_pos = self._get_init_pos(world)
		if not random_choice and len(self.worlds) != 1:
			self.i_curr_world += 1
		return Scenario(world, view_pos)

	def _load_worlds(self, path):
		""" load the images from path as worlds """
		dir_list = os.listdir(path)
		for f in dir_list:
			if f.endswith(self.WOLRD_IMG_FORMAT):
				ep = int(f.split(".")[0])
				f = path + "/" + f
				img = cv2.imread(f)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img = np.asarray(img, dtype=theano.config.floatX)
				img /= np.max(img)  # renorm to 1 as max
				self.worlds[ep] = img
		if self.path_init_file is not None:
			self.init_poses = self._get_init_pos_from_file(self.path_init_file)

	def _get_init_pos(self, world):
		if self.path_init_file is None:
			target_pos = self._get_target_pos(world)
			pos = None
			view = None
			# when required find initial view where path is visible
			while view is None or not self.is_view_valid(view):
				pos = self.sampler.sample_init_pos(self.bbox, target_pos)
				view = self.get_view(world, pos)
		else:
			pos = self.init_poses[self.i_curr_world]
		return pos

	def _get_init_pos_from_file(self, path):
		poses = {}
		_file = open(path)
		_file.next()  # skip header line
		for line in _file:
			vals = line.split("\t")
			epoch = int(vals[0])
			pos = Point(int(vals[1]), int(vals[2]))
			poses[epoch] = pos
		return poses

	def _get_target_pos(self, world):
		y = int(np.mean(np.where(world == self.TARGET_VALUE)[0]))
		x = int(np.mean(np.where(world == self.TARGET_VALUE)[1]))
		return Point(x, y)

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
		return len(np.where(view == self.TARGET_VALUE)[0])
