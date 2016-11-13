from abc import ABCMeta, abstractmethod


# abstract class for visualizing state - action course
class Visualizer(metaclass=ABCMeta):

	# visualizes just one given state
	@abstractmethod
	def visualize_state(self, state):
		pass

	# visualizes entire course of actions from given first state and actions
	@abstractmethod
	def visualize_course_of_action(self, first_state, actions):
		pass