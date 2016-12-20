from __future__ import division
import os, time
from matplotlib import pyplot as plt
import numpy as np
from sft.Actions import Actions


class Evaluator(object):
	"""visualizes the results, paths and parameters"""

	# TODO: sollte er alles aus dem Logger holen, damit wir es nur dort ändern müssen (VORSICHT, wenn wir es später ändern kriegen wir Probleme beim Einlesen

	INIT_STATES_FILE_NAME = "init_states.tsv"

	def __init__(self, exp_path, world_dir_name, agent_dict):
		self.agents_dict = agent_dict
		self.exp_path = exp_path
		self.world_dir_name = world_dir_name

	def _load_all_worlds_and_init_states(self):
		pass

	def _get_epoch_from_world_init_file_name(self, filename):
		# format is at the moment epoch22_worldstate.png
		return int(filename.split("_")[0].strip("epoch"))

	def plot_results(self, agents_dict, sliding_mean_window_size):
		pass

	def plot_paths(self, agent_keys, plot_every_kth_epoch, num_plot_paths):
		pass


# TODO: ALTER KREMPEL VON HIERAUS, NEU MACHEN/ EINARBEITEN

	def visualize_current_state(self, world_state, i, j, n, m):
		""" used to just plot and show the current state in the world_state (e.g. used to test simulators) """
		plt.clf()
		# plotting view window
		win_x = np.array([j, j, j + m, j + m, j])
		win_y = np.array([i, i + n, i + n, i, i])
		print("x={} y={}".format(win_x, win_y))
		plt.plot(win_x, win_y, 'r:')
		# plot world-frame
		plt.imshow(world_state, cmap="gray", alpha=0.8, interpolation='none')
		if self.plt_is_shown:
			plt.draw()
			plt.pause(0.001)
		else:
			self.plt_is_shown = True
			plt.ion()
			plt.show()
			plt.draw()
			plt.pause(0.001)

# TODO: momentan wie es hier ist nimmt er die obere linke Ecke als state, wir haben aber nun Mitte des Views als State Position
	def visualize_course_of_action(self, world_state, first_i, first_j, grid_n, grid_m, actions, title=None,
								   image_name="agent_path"):
		""" plots a course of actions beginning from a certain first state """
		n, m = grid_n, grid_m
		x = first_j + m // 2
		y = first_i + n // 2
		xx = np.array(x)
		yy = np.array(y)
		for ac in actions:
			(x, y) = self._get_new_xy(x, y, ac)
			xx = np.append(xx, x)
			yy = np.append(yy, y)
		# fig = plt.figure()
		plt.plot(xx, yy, 'b-', xx[-1], yy[-1], 'ro', xx[0], yy[0], 'go')
		# plotting starting view window
		first_win_x = np.array([first_j, first_j, first_j + m, first_j + m, first_j])
		first_win_y = np.array([first_i, first_i + n, first_i + n, first_i, first_i])
		plt.plot(first_win_x, first_win_y, 'r:')
		# plotting final view window
		mid_m = m // 2
		mid_n = n // 2
		final_win_x = np.array([xx[-1] - mid_m, xx[-1] - mid_m, xx[-1] + mid_m, xx[-1] + mid_m, xx[-1] - mid_m])
		final_win_y = np.array([yy[-1] - mid_n, yy[-1] + mid_n, yy[-1] + mid_n, yy[-1] - mid_n, yy[-1] - mid_n])
		plt.plot(final_win_x, final_win_y, 'r:')
		# plot world-frame
		plt.imshow(world_state, cmap="gray", alpha=0.8, interpolation='none')
		# remove x & y axis ticks
		plt.xticks([])
		plt.yticks([])
		if title is not None:
			plt.title(title)
		# save and clear figure
		image_save_path = "tmp/paths/" + image_name + ".png"
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