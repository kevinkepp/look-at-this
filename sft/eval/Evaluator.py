from __future__ import division
import os, time, cv2, ast, imp
import numpy as np
from sft.Actions import Actions
from sft import Point, Size

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib import pyplot as plt

class Evaluator(object):
	"""visualizes the results, paths and parameters"""

	INIT_STATES_FILE_NAME = "init_states.tsv"
	WORLD_INIT_LOGS = "world_init_logs"
	ACTIONS_FILE_NAME = "actions.tsv"
	EVAL_OUTPUT_DIR_NAME = "evaluation"
	PATHS_OUTPUT_DIR_NAME = "paths"
	WORLD_CONFIG_NAME = "world.py"
	RESULTS_FILE_NAME = "results.tsv"
	PARAMETER_LOG_FOLDER = "parameter_logs"
	RESULTS_PLOT_FILE_NAME = "results.png"
	PARAMETER_FILE_SUFFIX = ".tsv"

	def __init__(self, exp_path, world_dir_name, agent_dict):
		if not os.path.exists(exp_path):
			raise AttributeError("Experiment path %s does not exist" % exp_path)
		self.agents_dict = agent_dict
		self.exp_path = exp_path
		self.world_dir_name = world_dir_name
		self._create_folder(os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME))

		world_config_path = os.path.join(self.exp_path, self.world_dir_name, self.WORLD_CONFIG_NAME)
		self.view_size = self._get_view_size(world_config_path)

		self.worlds, self.init_poses = self._get_all_worlds_and_init_states()

	def _get_view_size(self, path):
		# hacky way to access the view size from the python config file
		_file = open(path, 'r')
		for line in _file:
			if line.startswith("view_size"):
				view_size_str = line.split("=")[1].strip().strip("Size(").strip(")").split(",")
				w, h = int(view_size_str[0]), int(view_size_str[1])
		return Size(w, h)

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
		_, valss = self._load_tsv_file_with_headline(actions_path)
		for vals in valss:
			epoch = int(vals[0])
			actions = ast.literal_eval(vals[1])
			actionss[epoch] = actions
		return actionss

	def _load_tsv_file_with_headline(self, path):
		_file = open(path)
		headline = _file.readline().split("\t")
		vals = []
		for line in _file:
			if line is not "":
				vals.append(line.split("\t"))
		return headline, vals

	def _get_q_values(self, path):
		_, vals = self._load_tsv_file_with_headline(path)
		qs = self._extract_qs_from_tsv_format(vals)
		epochs = np.array(vals)[:, 0]
		cur_epoch = 0
		i_last_epoch = 0
		q_dict = {}
		for i in range(len(epochs)):
			if epochs[i] != cur_epoch or i == len(epochs) - 1 :
				q_dict[int(cur_epoch)] = qs[i_last_epoch:i, :]
				cur_epoch = epochs[i]
				i_last_epoch = i
		return q_dict

	def _extract_qs_from_tsv_format(self, vals):
		qs = np.zeros([len(vals), 4])
		for i in range(len(vals)):
			v = vals[i][1]
			v = v.strip("[[").strip("]]\n").split(" ")
			v = filter(None, v)  # deletes empty strings, which are in there due to split
			for j in range(4):
				qs[i, j] = float(v[j])
		return qs

	def plot_qs(self, agent_keys, q_file):
		for agent_key in agent_keys:
			agent_qs_path = os.path.join(self.exp_path, self.agents_dict[agent_key], self.PARAMETER_LOG_FOLDER, q_file)
			_, vals = self._load_tsv_file_with_headline(agent_qs_path)
			qs = self._extract_qs_from_tsv_format(vals)
			# plot
			qs_max = np.max(qs, axis=1)
			qs_min = np.min(qs, axis=1)
			plt.figure(figsize=(30, 5), dpi=80)
			plt.plot(qs_max, 'b-', alpha=0.8, label="q-max")
			plt.plot(qs_min, 'r-', alpha=0.8, label="q-min")
			plt.plot(qs_max - qs_min, 'g-', alpha=0.8, label="q-max - q-min")
			plt.legend(loc="best")
			plt.ylabel("q for each step")
			plt.xlabel("step counter over all epochs")
			name = agent_key + "_q_all.png"
			save_path = os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME, name)
			plt.savefig(save_path)
			plt.close()

	def plot_one_value_parameter(self, agent_keys, parameter_file_name):
		parameter = parameter_file_name.strip(self.PARAMETER_FILE_SUFFIX)
		for agent_key in agent_keys:
			agent_qs_path = os.path.join(self.exp_path, self.agents_dict[agent_key], self.PARAMETER_LOG_FOLDER, parameter_file_name)
			_, valss = self._load_tsv_file_with_headline(agent_qs_path)
			epochs = []
			p = []
			for i in range(len(valss)):
				epochs.append(int(valss[i][0]))
				p.append(float(valss[i][1]))
			# plot
			plt.plot(epochs, p, 'b-')
			plt.ylabel(parameter)
			plt.xlabel("epochs")
			name = agent_key + "_" + parameter + ".png"
			save_path = os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME, name)
			plt.savefig(save_path)
			plt.close()


	def plot_paths(self, PLOT_EVERY_KTH_EPOCH, NUM_PLOT_PATHS_IN_ROW, q_file_name, text_every_kth):
		path = os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME, self.PATHS_OUTPUT_DIR_NAME)
		self._create_folder(path)  # create overall path-plot-folder
		agent_keys = self.agents_dict.keys()
		for agent_key in agent_keys:
			agent_dir = self.agents_dict[agent_key]
			agent_log_path = os.path.join(self.exp_path, agent_dir)
			if not os.path.exists(agent_log_path):
				print("Agent directory {0} not found".format(agent_dir))
				continue
			# get corresponding q-values
			q_file_path = os.path.join(agent_log_path, self.PARAMETER_LOG_FOLDER, q_file_name)
			qs_dict = self._get_q_values(q_file_path)
			# plotting
			plot_path = os.path.join(path, agent_dir)
			self._create_folder(plot_path)  # create path plot folder for each agent
			actionss = self._get_actions(agent_log_path)
			for epoch in actionss.keys():
				if epoch % PLOT_EVERY_KTH_EPOCH < NUM_PLOT_PATHS_IN_ROW:
					qs_of_epoch = qs_dict[epoch]
					w, h = self.view_size.w, self.view_size.h
					img_save_path = os.path.join(plot_path, str(epoch).zfill(5))
					self._visualize_course_of_action(self.worlds[epoch], self.init_poses[epoch], w, h, actionss[epoch],  text_every_kth)
					# save and clear figure
					plt.savefig(img_save_path + "_path.png", bbox_inches='tight')
					plt.close()
					self._visualize_qs_for_one_epoch(qs_of_epoch, text_every_kth)
					# save and clear figure
					plt.savefig(img_save_path + "_qs.png", bbox_inches='tight')
					plt.close()

	def animate_epoch(self, agent_key, epoch, q_file_name=None):
		# stuff for the animation
		FFMpegWriter = manimation.writers['ffmpeg']
		metadata = dict(title='epoch'+str(epoch), artist='DQ-Agent',
						comment='gogogo you can make it!')
		writer = FFMpegWriter(fps=2, metadata=metadata)
		# load world and init, actions, qs
		agent_dir = self.agents_dict[agent_key]
		agent_log_path = os.path.join(self.exp_path, agent_dir)
		actionss = self._get_actions(agent_log_path)
		actions = actionss[epoch]
		fig = plt.figure(figsize=(12, 12))
		path = os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME, self.PATHS_OUTPUT_DIR_NAME, agent_dir)
		movie_path = os.path.join(path, str(epoch).zfill(5) + "_" + agent_key + ".mp4")
		with writer.saving(fig, movie_path, len(actions)):
			# here the frame is create with matplotlib
			w, h = self.view_size.w, self.view_size.h
			self._visualize_course_of_action(self.worlds[epoch], self.init_poses[epoch], w, h, actions, movie_writer=writer)
		plt.close()

	def _visualize_course_of_action(self, world_state, init_pos, width, height, actions, text_every_kth=10, movie_writer=None):
		""" plots a course of actions beginning from a certain first state """
		x = init_pos.x
		y = init_pos.y
		xx = np.array(x)
		yy = np.array(y)
		mid_m = width/2.
		mid_n = height/2.
		# actions to x and y coordinates
		for ac in actions:
			(x, y) = self._get_new_xy(x, y, ac)
			xx = np.append(xx, x)
			yy = np.append(yy, y)
		# plot world-frame
		plt.imshow(world_state, cmap="gray", alpha=0.8, interpolation='none')
		# remove x & y axis ticks
		plt.xticks([])
		plt.yticks([])
		# path plot
		if movie_writer is None:
			# ration for plots
			plt.figure(figsize=(12, 12))
			self._plot_actions_as_path(xx, yy, text_every_kth)
			# plot start and end point
			plt.plot(xx[-1], yy[-1], 'ro', xx[0], yy[0], 'go')  # plt.plot(xx, yy, 'b-', xx[-1], yy[-1], 'ro', xx[0], yy[0], 'go')
			# plotting starting view window
			first_win_x = np.array([xx[0] - mid_m, xx[0] - mid_m, xx[0] + mid_m, xx[0] + mid_m, xx[0] - mid_m])
			first_win_y = np.array([yy[0] - mid_n, yy[0] + mid_n, yy[0] + mid_n, yy[0] - mid_n, yy[0] - mid_n])
			plt.plot(first_win_x, first_win_y, 'r:')
			# plotting final view window
			final_win_x = np.array([xx[-1] - mid_m, xx[-1] - mid_m, xx[-1] + mid_m, xx[-1] + mid_m, xx[-1] - mid_m])
			final_win_y = np.array([yy[-1] - mid_n, yy[-1] + mid_n, yy[-1] + mid_n, yy[-1] - mid_n, yy[-1] - mid_n])
			plt.plot(final_win_x, final_win_y, 'r:')
		else:
			plt.plot(xx, yy, 'b-')  # entire path
			p_d, = plt.plot(xx[:2], yy[:2], 'r-', linewidth=3)  # direction
			p_a, = plt.plot(xx[0], yy[0], 'go')  # agent
			win_x = np.array([xx[0] - mid_m, xx[0] - mid_m, xx[0] + mid_m, xx[0] + mid_m, xx[0] - mid_m])
			win_y = np.array([yy[0] - mid_n, yy[0] + mid_n, yy[0] + mid_n, yy[0] - mid_n, yy[0] - mid_n])
			p_v, = plt.plot(win_x, win_y, 'g-', linewidth=3)  # view
			movie_writer.grab_frame()
			for i_ac in range(1,len(actions)):
				p_d.set_data(xx[i_ac:i_ac + 2], yy[i_ac:i_ac + 2])  # direction
				p_a.set_data(xx[i_ac], yy[i_ac])  # agent
				win_x = np.array(
					[xx[i_ac] - mid_m, xx[i_ac] - mid_m, xx[i_ac] + mid_m, xx[i_ac] + mid_m, xx[i_ac] - mid_m])
				win_y = np.array(
					[yy[i_ac] - mid_n, yy[i_ac] + mid_n, yy[i_ac] + mid_n, yy[i_ac] - mid_n, yy[i_ac] - mid_n])
				p_v.set_data(win_x, win_y)  # view
				movie_writer.grab_frame()

	def _plot_actions_as_path(self, xx, yy, text_every_kth):
		col_flag = 0
		col_list = ['r', 'g', 'y', 'm', 'c']
		# plot color order for orientation
		x = 0
		for c in col_list:
			plt.plot([x, x+1], [0, 0], c + '-')
			x += 1
		# plot actions
		for i in range(len(xx)-1):
			plt.plot(xx[i:i+2], yy[i:i+2], col_list[col_flag] + '-')
			if col_flag != 4:
				col_flag += 1
			else:
				col_flag = 0
			# text
			if i % text_every_kth == 0:
				plt.text(xx[i], yy[i], str(i), color="white")

	def _visualize_qs_for_one_epoch(self, qs, text_every_kth):
		# q-plot every action dot
		plt.figure(figsize=(24, 4))
		gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], width_ratios=[1, 1])
		plt.subplot(gs[0])
		q_colors = ["r", "g", "b", "k"]
		for i_ac in range(len(Actions.all)):
			plt.plot(qs[:, i_ac], q_colors[i_ac] + ".", label=Actions.name_dict[i_ac])
			plt.xlim([-1, len(qs)])
		plt.gca().xaxis.grid(b=True, which='major', linestyle='--', alpha=0.6)
		plt.gca().xaxis.grid(b=True, which='minor', linestyle='--', alpha=0.3)
		xticks_major = MultipleLocator(text_every_kth)
		xticks_minor = MultipleLocator(2)
		plt.gca().xaxis.set_major_locator(xticks_major)
		plt.gca().xaxis.set_minor_locator(xticks_minor)
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center', ncol=len(Actions.all))
		# plot difference max q to second highest
		plt.subplot(gs[1])
		qs_sorted = np.sort(qs, axis=1)
		q_diff = qs_sorted[:, -1] - qs_sorted[:, -2]
		plt.plot(q_diff, 'b-')
		plt.xlim([-1, len(qs)])
		plt.gca().xaxis.grid(b=True, which='major', linestyle='--', alpha=0.6)
		plt.gca().xaxis.grid(b=True, which='minor', linestyle='--', alpha=0.3)
		xticks_major = MultipleLocator(text_every_kth)
		xticks_minor = MultipleLocator(2)
		plt.gca().xaxis.set_major_locator(xticks_major)
		plt.gca().xaxis.set_minor_locator(xticks_minor)
		plt.gca().xaxis.set_major_formatter(NullFormatter())
		# formating
		plt.tight_layout()

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

	def plot_results(self, agent_keys, sliding_window_mean):
		successes = []
		stepss = []
		names = []
		epochs = 0
		for agent_key in agent_keys:
			# load results
			agent_res_path = os.path.join(self.exp_path, self.agents_dict[agent_key], self.RESULTS_FILE_NAME)
			epoch, success, amt_act_taken = self._load_results(agent_res_path)
			# append results and name for later plotting
			successes.append(success)
			stepss.append(amt_act_taken)
			names.append(agent_key)
			epochs = epoch[-1]
		# plot the results
		fig, ax_success = plt.subplots()
		ax_steps = ax_success.twinx()
		title = "Training over {0} epochs (steps avg over last {1} epochs)".format(epochs, sliding_window_mean)
		plt.title(title)
		ax_steps.set_xlabel("epochs")
		ax_steps.set_ylabel("# steps taken")
		ax_success.set_ylabel("success-rate")
		ax_success.grid(True)
		ax_steps.grid(True, alpha=0.3)
		ax_success.set_xlim((sliding_window_mean - 2, epochs + 1))
		ax_success.set_ylim((-0.02, 1.02))
		for name, success, steps in zip(names, successes, stepss):
			self._plot_res_one_agent(ax_success, ax_steps, success, steps, name, epochs, sliding_window_mean)
		ax_success.legend(loc='upper right')
		filepath = os.path.join(self.exp_path, self.EVAL_OUTPUT_DIR_NAME, self.RESULTS_PLOT_FILE_NAME)
		plt.savefig(filepath, bbox_inches='tight')
		plt.close()
		# save figure

	def _load_results(self, res_path):
		headline, valss = self._load_tsv_file_with_headline(res_path)
		epochs, successes, amt_act_taken = [], [], []
		for vals in valss:
			epochs.append(int(vals[0]))
			successes.append(int(vals[1]))
			amt_act_taken.append(int(vals[2]))
		return epochs, successes, amt_act_taken

	def _plot_res_one_agent(self, ax_success, ax_steps, success, steps, name, epochs, window_size):
		np_success = self._movingaverage(np.array(success), window_size)
		np_steps = self._movingaverage(np.array(steps), window_size)
		x = np.arange(window_size - 1, epochs + 1)
		ax_success.plot(x, np_success, '-', label=name)
		ax_steps.plot(x, np_steps, ':')

	def _movingaverage(self, values, window):
		weights = np.repeat(1.0, window) / window
		mav = np.convolve(values, weights, 'valid')
		return mav
