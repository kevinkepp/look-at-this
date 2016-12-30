from __future__ import division
import os, time, cv2, ast, imp
from matplotlib import pyplot as plt
import numpy as np
from sft.Actions import Actions
from sft import Point


class Evaluator(object):
	"""visualizes the results, paths and parameters"""

	INIT_STATES_FILE_NAME = "init_states.tsv"
	WORLD_INIT_LOGS = "world_init_logs"
	ACTIONS_FILE_NAME = "actions.tsv"
	EVAL_OUTPUT_DIR_NAME = "evaluation"
	PATHS_OUTPUT_DIR_NAME = "paths"
	WORLD_CONFIG_NAME = "world.py"

	def __init__(self, exp_path, world_dir_name, agent_dict):
		self.agents_dict = agent_dict
		self.exp_path = exp_path
		self.world_dir_name = world_dir_name
		self._create_folder(os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME))

		results_dir = os.path.join(self.exp_path, self.world_dir_name)
		self.view_size = self._get_view_size()

	def _get_view_size(self):
		# TODO: das herausfinden der View Size aus dem File funktioniert noch nicht (import geht nicht wegen problemen mit Logger - evtl probieren einfach als Datei zeilenweise einzulesen und dann nach Keyword View-size zu suchen
		return self.view_size

	def _create_folder(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def _get_all_worlds_and_init_states(self):
		worlds = {}
		files = os.listdir(os.path.join(self.exp_path, self.world_dir_name, self.WORLD_INIT_LOGS))
		for _file in files:
			if _file.endswith(".png"):
				world_image_path = os.path.join(self.exp_path, self.world_dir_name, self.WORLD_INIT_LOGS, _file)
				world_image = cv2.imread(world_image_path)
				epoch = int(_file[len("epoch"):_file.index('_')])
				worlds[epoch] = world_image
		init_poses = self._get_init_poses()
		return worlds, init_poses

	def _get_init_poses(self):
		poses = {}
		init_pos_path = os.path.join(self.exp_path, self.world_dir_name, self.WORLD_INIT_LOGS, self.INIT_STATES_FILE_NAME)
		_file = open(init_pos_path)
		_file.next()  # skip header line
		for line in _file:
			vals = line.split("\t")
			epoch = int(vals[0])
			pos = Point(int(vals[1]), int(vals[2]))
			poses[epoch] = pos
		return poses

	def _get_actions(self, agent_path):
		actionss = {}
		actions_path = os.path.join(agent_path, self.ACTIONS_FILE_NAME)
		_file = open(actions_path)
		_file.next()  # skip header line
		for line in _file:
			vals = line.split("\t")
			epoch = int(vals[0])
			actions = ast.literal_eval(vals[1])
			actionss[epoch] = actions
		return actionss

	def plot_paths(self, PLOT_EVERY_KTH_EPOCH, NUM_PLOT_PATHS_IN_ROW):
		path = os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME, self.PATHS_OUTPUT_DIR_NAME)
		self._create_folder(path)  # create overall path-plot-folder
		worlds, init_poses = self._get_all_worlds_and_init_states()
		agent_keys = self.agents_dict.keys()
		for agent_key in agent_keys:
			agent_dir = self.agents_dict[agent_key]
			plot_path = os.path.join(path, agent_dir)
			self._create_folder(plot_path)  # create path plot folder for each agent
			action_path = os.path.join(self.exp_path, agent_dir)
			actionss = self._get_actions(action_path)
			for epoch in actionss.keys():
				if epoch % PLOT_EVERY_KTH_EPOCH < NUM_PLOT_PATHS_IN_ROW:
					w, h = self.view_size.w, self.view_size.h
					img_save_path = os.path.join(plot_path, str(epoch).zfill(5) + ".png")
					self._visualize_course_of_action(worlds[epoch], init_poses[epoch], w, h, actionss[epoch], img_save_path)


	def plot_results(self, agent_keys, sliding_mean_window_size):
		for agent_key in agent_keys:
			pass

	def _visualize_course_of_action(self, world_state, init_pos, width, height, actions, image_save_path):
		""" plots a course of actions beginning from a certain first state """
		x = init_pos.x
		y = init_pos.y
		first_i = y - int(height/2)
		first_j = x - int(width/2)
		xx = np.array(x)
		yy = np.array(y)

		mid_m = int(width/2)
		mid_n = int(height/2)

		for ac in actions:
			(x, y) = self._get_new_xy(x, y, ac)
			xx = np.append(xx, x)
			yy = np.append(yy, y)

		# fig = plt.figure()
		plt.plot(xx, yy, 'b-', xx[-1], yy[-1], 'ro', xx[0], yy[0], 'go')
		# plotting starting view window
		first_win_x = np.array([xx[0] - mid_m, xx[0] - mid_m, xx[0] + mid_m, xx[0] + mid_m, xx[0] - mid_m])
		first_win_y = np.array([yy[0] - mid_n, yy[0] + mid_n, yy[0] + mid_n, yy[0] - mid_n, yy[0] - mid_n])
		plt.plot(first_win_x, first_win_y, 'r:')
		# plotting final view window
		final_win_x = np.array([xx[-1] - mid_m, xx[-1] - mid_m, xx[-1] + mid_m, xx[-1] + mid_m, xx[-1] - mid_m])
		final_win_y = np.array([yy[-1] - mid_n, yy[-1] + mid_n, yy[-1] + mid_n, yy[-1] - mid_n, yy[-1] - mid_n])
		plt.plot(final_win_x, final_win_y, 'r:')
		# plot world-frame
		plt.imshow(world_state, cmap="gray", alpha=0.8, interpolation='none')
		# remove x & y axis ticks
		plt.xticks([])
		plt.yticks([])
		# save and clear figure
		plt.savefig(image_save_path)
		# plt.clf()
		plt.close()

	def _get_new_xy(self, x, y, ac):
		""" calculates the new position of the goal after a action """
		if ac == Actions.UP:
			y -= 1
		elif ac == Actions.DOWN:
			y += 1
		elif ac == Actions.RIGHT:
			x += 1
		elif ac == Actions.LEFT:
			x -= 1
		return x, y

	# TODO: ALTER KREMPEL VON HIERAUS, NEU MACHEN/ EINARBEITEN

	def plot_results(self, names, results, epochs, window_size, params):
		""" plot the results in a plot with two y axis, one the success-rate and the second
		the difference between steps-taken and min-necessary steps"""
		fig, ax_success = plt.subplots()
		ax_steps = ax_success.twinx()
		title = "Training over {0} epochs (steps avg over last {1} epochs)".format(epochs, window_size)
		plt.title(title)
		ax_steps.set_xlabel("epochs")
		ax_steps.set_ylabel("steps taken - min steps")
		ax_success.set_ylabel("success-rate")
		ax_success.grid(True)
		ax_steps.grid(True, alpha=0.3)
		ax_success.set_xlim((window_size - 1, epochs + 1))
		ax_success.set_ylim((-0.02, 1.02))

		for name, result in zip(names, results):
			self._plot_res_one_agent(ax_success, ax_steps, result, name, window_size)

		ax_success.legend(loc='center right')
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		para = "_epochs=%d" % epochs
		for n, v in sorted(params.items()):
			para += "_{0}={1}".format(n, v)
		filename = "tmp/plots/" + timestamp + para + ".png"
		plt.savefig(filename, bbox_inches='tight')
		plt.show()

	def _plot_res_one_agent(self, ax_success, ax_steps, results, name, window_size):
		results = np.array(results)
		step_diff = self._movingaverage(results[:, 1] - results[:, 2], window_size)
		success = self._movingaverage(results[:, 0], window_size)
		x = np.arange(window_size, len(results) + 1)
		ax_success.plot(x, success, '-', label=name)
		ax_steps.plot(x, step_diff, '--')

	def _movingaverage(self, values, window):
		weights = np.repeat(1.0, window) / window
		mav = np.convolve(values, weights, 'valid')
		return mav