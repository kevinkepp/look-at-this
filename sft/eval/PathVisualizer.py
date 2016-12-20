from eval.SimpleVisualize import PathAndResultsPlotter
import os
import cv2

from sft import Point


class PathVisualizer(object):
	RESULTS_DIR = "tmp/paths/"
	CONFIG_DIR = "config-sample/"
	WORLD_CONFIG_NAME = "world.py"
	WORLD_INIT_DIR = "world_init_logs/"
	WORLD_INIT_NAME = "world_init_logs"
	RESULTS_NAME = "results.tsv"

	visualizer = PathAndResultsPlotter()

	def visualize_paths(self, results_dir):
		world_config = self.get_world_config(results_dir)
		worlds = self.get_world_images(results_dir)
		poss = self.get_init_poss(results_dir)
		actionss = self.get_actionss(results_dir)
		_dir = results_dir.split("/")[-1]
		if not os.path.exists(self.RESULTS_DIR + _dir):
			os.makedirs(self.RESULTS_DIR + _dir)
		# print 100 paths
		for i in range(0, len(worlds), len(worlds) / 20):
			world = worlds[i]
			pos = poss[i]
			actions = actionss[i]
			name = _dir + "/" + "epoch" + str(i)
			# calc start of view because mid is given
			start_pos = pos - Point(world_config.view_size.w / 2, world_config.view_size.h / 2)
			self.visualizer.visualize_course_of_action(world, start_pos.x, start_pos.y,
													   world_config.view_size.w, world_config.view_size.h,
													   actions, image_name=name)

	def get_world_config(self, results_dir):
		import imp

		path = results_dir + "/" + self.CONFIG_DIR + self.WORLD_CONFIG_NAME
		return imp.load_source(self.WORLD_CONFIG_NAME.split(".")[0], path)

	def get_world_images(self, results_dir):
		worlds = {}
		files = os.listdir(results_dir + "/" + self.WORLD_INIT_DIR)
		for _file in files:
			if _file.endswith(".png"):
				world_image_path = results_dir + "/" + self.WORLD_INIT_DIR + _file
				world_image = cv2.imread(world_image_path)
				epoch = int(_file[len("epoch"):_file.index('_')])
				worlds[epoch] = world_image
		return worlds

	def get_init_poss(self, results_dir):
		poss = {}
		init_pos_path = results_dir + "/" + self.WORLD_INIT_DIR + self.WORLD_INIT_NAME
		_file = open(init_pos_path)
		_file.next()  # skip header line
		for line in _file:
			vals = line.split("\t")
			epoch = int(vals[0])
			pos = Point(int(vals[1]), int(vals[2]))
			poss[epoch] = pos
		return poss

	def get_actionss(self, results_dir):
		import ast

		actionss = {}
		_file = open(results_dir + "/" + self.RESULTS_NAME)
		_file.next()  # skip header line
		for line in _file:
			vals = line.split("\t")
			epoch = int(vals[0])
			actions = ast.literal_eval(vals[2])
			actionss[epoch] = actions
		return actionss
