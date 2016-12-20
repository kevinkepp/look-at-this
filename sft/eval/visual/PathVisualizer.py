from eval.SimpleVisualize import PathAndResultsPlotter
import os
import cv2

from sft import Point


class PathVisualizer(object):
	RESULTS_DIR = "tmp/paths/"
	WORLD_DIR = "world/"
	WORLD_CONFIG_NAME = "world.py"
	WORLD_INIT_DIR = "world_init_logs/"
	WORLD_INIT_NAME = "init_states.tsv"
	ACTIONS_NAME = "actions.tsv"

	visualizer = PathAndResultsPlotter()

	def visualize_paths(self, results_dir, exp_module_name, agent_name):
		world_config = self.get_world_config(results_dir, exp_module_name)
		worlds = self.get_world_images(results_dir)
		poss = self.get_init_poss(results_dir)
		actionss = self.get_actionss(results_dir, agent_name)
		_dir = os.path.join(results_dir.split("/")[-1], agent_name)
		if not os.path.exists(self.RESULTS_DIR + _dir):
			os.makedirs(self.RESULTS_DIR + _dir)
		# print 100 paths
		for i in range(0, len(actionss), len(actionss) / 20):
			world = worlds[i]
			pos = poss[i]
			actions = actionss[i]
			name = os.path.join(_dir, "epoch" + str(i))
			# calc start of view because mid is given
			start_pos = pos - Point(world_config.view_size.w / 2, world_config.view_size.h / 2)
			self.visualizer.visualize_course_of_action(world, start_pos.x, start_pos.y,
													   world_config.view_size.w, world_config.view_size.h,
													   actions, image_name=name)

	def get_world_config(self, results_dir, exp_module_name):
		import imp

		module_name = exp_module_name + ".world"
		path = os.path.join(results_dir, self.WORLD_DIR, self.WORLD_CONFIG_NAME)
		return imp.load_source(module_name, path)

	def get_world_images(self, results_dir):
		worlds = {}
		files = os.listdir(os.path.join(results_dir, "world", self.WORLD_INIT_DIR))
		for _file in files:
			if _file.endswith(".png"):
				world_image_path = os.path.join(results_dir, "world", self.WORLD_INIT_DIR, _file)
				world_image = cv2.imread(world_image_path)
				epoch = int(_file[len("epoch"):_file.index('_')])
				worlds[epoch] = world_image
		return worlds

	def get_init_poss(self, results_dir):
		poss = {}
		init_pos_path = os.path.join(results_dir, self.WORLD_DIR, self.WORLD_INIT_DIR, self.WORLD_INIT_NAME)
		_file = open(init_pos_path)
		_file.next()  # skip header line
		for line in _file:
			vals = line.split("\t")
			epoch = int(vals[0])
			pos = Point(int(vals[1]), int(vals[2]))
			poss[epoch] = pos
		return poss

	def get_actionss(self, results_dir, agent_name):
		import ast

		actionss = {}
		_file = open(os.path.join(results_dir, agent_name, self.ACTIONS_NAME))
		_file.next()  # skip header line
		for line in _file:
			vals = line.split("\t")
			epoch = int(vals[0])
			actions = ast.literal_eval(vals[1])
			actionss[epoch] = actions
		return actionss
