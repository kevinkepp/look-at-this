from lat.Environment import Environment

from copy import copy
from enum import Enum
import numpy as np

class Actions(Enum):
	up = 0
	down = 1
	left = 2
	right = 3

	@staticmethod
	def all():
		return [a for a in Actions]


class SimpleMatrixSimulator(Environment):

	world_factor = 3 # the "world" has a size factor x grid_dims

	"""simulates an image frame environment for a learning agent"""
	def __init__(self, agent, reward, grid_n, grid_m=1, orientation=0, max_steps=1000, visualizer=None, bounded=True):
		self.agent = agent
		self.reward = reward
		self.visual = visualizer
		self.grid_dims = self._get_odd_dims(grid_n,grid_m)
		self.orientation = orientation
		self.max_steps = max_steps
		self.bounded = bounded

		self.state = None
		self.old_state = None
		self.all_states = None # d x n x m dimensional numpy array, with d states of size n x m in it - used for visualization later
		# world state from which state is extracted
		self.world_state = None
		# top left corner of the window
		self.i_world = 0 
		self.j_world = 0 

	def _get_odd_dims(self,n,m):
		"""setting dimensions of image so that it has uneven number of elements on both dims to be able to center at the middle """
		if n%2 == 0:
			n += 1
			print("Redefining dimension n to be ",n, "to have a middle pixel")
		if m%2 == 0:
			m += 1
			print("Redefining dimension m to be ",m, "to have a middle pixel")
		return (n,m)

	def _get_rand_matrix_state(self,dims):
		"""initialize a random state that is a matrix with zeros and one one in it at a random position """
		(n,m) = dims
		N = self.world_factor * n
		M = self.world_factor * m
		mid_N = N//2
		mid_M = M//2
		# create world-state
		self.world_state = np.zeros((N,M))
		self.world_state[mid_N,mid_M] = 1 
		i = np.random.randint(mid_N-n+1,mid_N+1)
		j = np.random.randint(mid_M-m+1,mid_M+1)
		# avoid generation in the middle
		if i+n//2 == mid_N and j+m//2 == mid_M:
		    i += np.random.choice([-1,1])
		self.i_world = i
		self.j_world = j
		return self.world_state[i:i+n,j:j+m]

	def _extract_state_from_world(self,i,j):
		(n,m) = self.grid_dims
		return self.world_state[i:i+n,j:j+m]

	def get_rand_heat_state(self):
		"""initialize the state to be a zero matrix with a gaussian distribution at a random position """
		pass

	def run_epoch(self, visible=False, trainingmode=False):
		self.state = self._get_rand_matrix_state(self.grid_dims)
		if visible:
			self.visual.visualize_state(self.state)
			self.first_state = copy(self.state) # stores first state for later visualization
		best = self.get_best_possible_steps()
		steps = []
		while len(steps) < self.max_steps:
			self.old_state = copy(self.state)
			action = self.agent.choose_action(self.state)
			steps.append(action)
			self.execute_action(action)
			if trainingmode:
				reward = self.reward.get_reward(self.old_state, self.state, self._is_oob())
				print("reward is ",reward) 
				self.agent.incorporate_reward(self.old_state, action, self.state, reward)
			if visible:
				self.visual.visualize_state(self.state)
				# self.all_states = np.concatenate((self.all_states, self.state[np.newaxis, :, :]), axis=0)
			if self._is_oob():
				return 0, steps, best
			elif self._at_goal(self.state):
				return 1, steps, best
		return 0, steps, best

	def run(self, epochs=1, visible=False, trainingmode=False):
		res = []
		for i in range(epochs):
			self.agent.new_epoch(i)
			r = self.run_epoch(visible, trainingmode)
			if visible:
				self.visual.visualize_course_of_action(self.first_state, r[1], image_name = "agent_path_" + str(i) + ".png")
			# show progress
			if epochs > 10 and i % int(epochs / 10) == 0:
				print("Epoch {0}/{1}".format(i, epochs))
			res.append(r)
		return res

	def _get_middle(self):
		""" returns middle of the state grid as tuple (mid_i, mid_j)"""
		(n,m) = self.grid_dims
		return (n // 2, m // 2)

	def _at_goal(self, state):
		""" returns True if at goal position and false otherwise """
		(mid_n, mid_m) = self._get_middle()
		return self.state[mid_n,mid_m] == 1

	def get_current_state(self):
		"""return the current state """
		return self.state

	# is out of bounds
	def _is_oob(self):
		return np.sum(self.state) == 0

	def execute_action(self, action):
		"""execute the action and therefore change the state """
		self.state = self._shift_image(action)

	def _shift_image(self,direction):
		"""actual state change for the matrix image variant """
		if Actions.up == direction:
			self.i_world -= 1
		elif Actions.right == direction:
			self.j_world += 1
		elif Actions.down == direction:	
			self.i_world += 1
		elif Actions.left == direction:
			self.j_world -= 1
		return self._extract_state_from_world(self.i_world, self.j_world)

	def calc_reward(self, state, old_state):
		"""receive a reward from the reward object """
		return self.reward.get_reward(state, old_state)

	def get_best_possible_steps(self):
		x, y = np.where(self.state == 1)
		mid_x = int(np.floor(self.grid_dims[0] / 2))
		mid_y = int(np.floor(self.grid_dims[1] / 2))
		return abs(mid_x - x[0]) + abs(mid_y - y[0])
