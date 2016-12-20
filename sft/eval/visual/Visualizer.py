from abc import ABCMeta, abstractmethod


# abstract class for visualizing state - action course
class Visualizer(object):
	__metaclass__ = ABCMeta

	# visualizes just one given state
	@abstractmethod
	def visualize_state(self, state):
		pass

	# visualizes entire course of actions from given first state and actions
	@abstractmethod
	def visualize_course_of_action(self, world_state, first_i, first_j, grid_n, grid_m, actions, title, image_name):
		pass
