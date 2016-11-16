from __future__ import division
from lat.Visualizer import Visualizer
from lat.Simulator import Actions

import numpy as np
import matplotlib.pyplot as plt


class PlotMatrix(Visualizer):
	""" Visualizes the states of a given matrix as state with one one in it as goal """

	def visualize_state(self, state):
		""" just plots current state in a simple print of matrix"""
		print(state)

	def visualize_course_of_action(self, first_state, actions, image_name="agent_path.png"):
		""" plots a course of actions beginning from a certain first state """
		n,m = first_state.shape
		y,x = np.where(first_state==1)
		x = x[0]
		y = y[0]
		X = np.array(x)
		Y = np.array(y)
		for ac in actions:
			(x,y) = self._get_new_xy(x,y,ac)
			X = np.append(X,x)
			Y = np.append(Y,y)
		plt.figure
		plt.plot(X,-Y,'b-',X[0],-Y[0],'ro',X[-1],-Y[-1],'rx')
		plt.xlim((-0.5,m-0.5))
		plt.ylim((-n+0.5,0.5))
		plt.xticks([])
		plt.yticks([])
		plt.savefig(image_name)
		plt.clf()

	def _get_new_xy(self, x, y, ac):
		""" calculates the new position of the goal after a action """
		if ac == Actions.up:
			y += 1
		elif ac == Actions.down:
			y -= 1
		elif ac == Actions.right:
			x -= 1
		elif ac == Actions.left:
			x += 1
		return (x,y)