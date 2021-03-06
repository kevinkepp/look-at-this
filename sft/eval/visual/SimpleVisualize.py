from __future__ import division

import time

import matplotlib.pyplot as plt
import numpy as np

from sft.Actions import Actions
from sft.eval.visual.Visualizer import Visualizer


class PathAndResultsPlotter(Visualizer):
	""" Visualizes the states of a given matrix as state with one one in it as goal """

	plt_is_shown = False

	def visualize_state(self, state):
		""" just plots current state in a simple print of matrix"""
		print(state)

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

	def visualize_course_of_action(self, world_state, first_x, first_y, grid_w, grid_h, actions, title=None,
								   image_name="agent_path"):
		""" plots a course of actions beginning from a certain first state """
		w, h = grid_w, grid_h
		x = first_x + w / 2
		y = first_y + h / 2
		xx = np.array(x)
		yy = np.array(y)
		for ac in actions:
			(x, y) = self._get_new_xy(x, y, ac)
			xx = np.append(xx, x)
			yy = np.append(yy, y)
		# fig = plt.figure()
		plt.plot(xx, yy, 'b-', xx[-1], yy[-1], 'ro', xx[0], yy[0], 'go')
		# plotting starting view window
		first_win_x = np.array([first_x, first_x + w, first_x + w, first_x, first_x])
		first_win_y = np.array([first_y, first_y, first_y + h, first_y + h, first_y])
		plt.plot(first_win_x, first_win_y, 'r:')
		# plotting final view window
		mid_x = w / 2
		mid_y = h / 2
		final_win_x = np.array([xx[-1] - mid_x, xx[-1] - mid_x, xx[-1] + mid_x, xx[-1] + mid_x, xx[-1] - mid_x])
		final_win_y = np.array([yy[-1] - mid_y, yy[-1] + mid_y, yy[-1] + mid_y, yy[-1] - mid_y, yy[-1] - mid_y])
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
		plt.clf()
		# plt.close()

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
