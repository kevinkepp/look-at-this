from lat.Visualizer import Visualizer
from lat.Simulator import Actions

import numpy as np
import matplotlib.pyplot as plt


class PlotMatrix(Visualizer):
	""" Visualizes the states of a given matrix as state with one one in it as goal """

	def visualize_state(self, state):
		""" just plots current state in a simple print of matrix"""
		print(state)

	def visualize_course_of_action(self, world_state, first_i, first_j, grid_n, grid_m, actions, title=None, image_name="agent_path"):
		""" plots a course of actions beginning from a certain first state """
		n, m = grid_n, grid_m
		x = first_j + m//2
		y = first_i + n//2
		xx = np.array(x)
		yy = np.array(y)
		for ac in actions:
			(x,y) = self._get_new_xy(x,y,ac)
			xx = np.append(xx,x)
			yy = np.append(yy,y)
		plt.figure
		plt.plot(xx, yy, 'b-', xx[-1], yy[-1], 'ro')
		# plotting starting view window
		first_win_x = np.array([first_j, first_j, first_j + m, first_j + m, first_j])
		first_win_y = np.array([first_i, first_i+n, first_i+n, first_i, first_i])
		plt.plot(first_win_x, first_win_y, 'r:')
		# plotting final view window
		mid_m = m//2
		mid_n = n//2
		final_win_x = np.array([xx[-1]-mid_m, xx[-1]-mid_m, xx[-1]+mid_m, xx[-1]+mid_m, xx[-1]-mid_m])
		final_win_y = np.array([yy[-1]-mid_n, yy[-1]+mid_n, yy[-1]+mid_n, yy[-1]-mid_n, yy[-1]-mid_n])
		plt.plot(final_win_x, final_win_y, 'r:')
		# plot world-frame
		plt.imshow(world_state, cmap="gray", alpha=0.8, interpolation='none')
		# remove x & y axis ticks
		plt.xticks([])
		plt.yticks([])
		if title is not None:
			plt.title(title)
		# save and clear figure
		image_save_path = "temp/"+image_name+".png"
		plt.savefig(image_save_path)
		plt.clf()

	def _get_new_xy(self, x, y, ac):
		""" calculates the new position of the goal after a action """
		if ac == Actions.up:
			y -= 1
		elif ac == Actions.down:
			y += 1
		elif ac == Actions.right:
			x += 1
		elif ac == Actions.left:
			x -= 1
		return x, y
