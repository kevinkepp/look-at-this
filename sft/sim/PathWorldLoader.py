import cv2, os, random
import numpy as np

from sft import Point, Rectangle, get_path_bbox
from sft.Scenario import Scenario
from sft.sim.ScenarioGenerator import ScenarioGenerator


class PathWorldLoader(ScenarioGenerator):

	WOLRD_IMG_FORMAT = ".png"
	TARGET_VALUE = 1.

	def __init__(self, logger, world_path, view_size, world_size, sampler, path_in_init_view=False, target_not_in_init_view=False):
		self.logger = logger
		self.view_size = view_size
		self.world_size = world_size
		self.bbox = get_path_bbox(world_size, view_size)
		self.sampler = sampler
		self.path_in_init_view = path_in_init_view
		self.target_not_in_init_view = target_not_in_init_view
		self.worlds = []  # used to store all loaded worlds
		self._load_worlds(world_path)

	def get_next(self):
		# choose a world
		if len(self.worlds) == 1:
			world = self.worlds[0]
		else:
			world = random.choice(self.worlds)
		# get init pos
		view_pos = self._get_init_pos(world)
		return Scenario(world, view_pos)

	def _load_worlds(self, path):
		""" load the images from path as worlds """
		dir_list = os.listdir(path)
		for f in dir_list:
			if f.endswith(self.WOLRD_IMG_FORMAT):
				f = path + "/" + f
				img = cv2.imread(f)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img = img / float(np.max(img))  # renorm to 1 as max
				self.worlds.append(img)

	def _get_init_pos(self, world):
		target_pos = self._get_target_pos(world)
		pos = None
		view = None
		# when required find initial view where path is visible
		while view is None or not self.is_view_valid(view):
			pos = self.sampler.sample_init_pos(self.bbox, target_pos)
			view = self.get_view(world, pos)
		return pos

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
