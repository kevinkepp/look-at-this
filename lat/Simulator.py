from lat.Environment import Environment

from copy import copy
# from enum import Enum
import numpy as np

# class Actions(Enum):
# 	up = 0
# 	down = 1
# 	left = 2
# 	right = 3
#
# 	@staticmethod
# 	def all():
# 		return [a for a in Actions]

def enum(**enums):
    return type('Enum', (), enums)

Actions = enum(up=0, down=1, left=2, right=3)
Actions.all = [Actions.up, Actions.down, Actions.left, Actions.right]


class SimpleMatrixSimulator(Environment):

	world_factor = 3 # the "world" has a size factor x grid_dims
	window_gen_factor = 0.75
	target = 1

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

	def _get_init_state(self,dims):
		"""initialize a random state that is a matrix with zeros and one one in it at a random position """
		(n,m) = dims
		N = self.world_factor * n
		M = self.world_factor * m
		mid_N = N//2
		mid_M = M//2
		# create world-state
		self.world_state = np.zeros((N,M))
		self.world_state[mid_N, mid_M] = 1
		i = np.random.randint(mid_N - n * self.window_gen_factor + 1, mid_N - n * (1 - self.window_gen_factor) + 1)
		j = np.random.randint(mid_M - m * self.window_gen_factor + 1, mid_M - m * (1 - self.window_gen_factor) + 1)
		# avoid generation in the middle
		if i+n//2 == mid_N and j+m//2 == mid_M:
			i += np.random.choice([-1,1])
		self.i_world = i
		self.j_world = j
		return self.world_state[i:i+n,j:j+m]

	def _extract_state_from_world(self,i,j):
		(n,m) = self.grid_dims
		return self.world_state[i:i+n,j:j+m]


	def _run_epoch(self, visible=False, trainingmode=False):
		self.state = self._get_init_state(self.grid_dims)
		if visible:
			self.visual.visualize_state(self.state)
			self.first_state = copy(self.state) # stores first state for later visualization
		best = self.get_best_possible_steps()
		steps = []
		while len(steps) < self.max_steps:
			self.old_state = copy(self.state)
			action = self.agent.choose_action(self.state)
			steps.append(action)
			self._execute_action(action)
			if trainingmode:
				reward = self.reward.get_reward(self.old_state, self.state, self._is_oob())
				#print("reward is ",reward)
				self.agent.incorporate_reward(self.old_state, action, self.state, reward)
			if visible:
				self.visual.visualize_state(self.state)
				# self.all_states = np.concatenate((self.all_states, self.state[np.newaxis, :, :]), axis=0)
			if self._is_oob():
				return 0, steps, best
			elif self._at_goal(self.state):
				return 1, steps, best
		return 0, steps, best

	def run_epoch(self, epoch_no=1, visible=False, trainingmode=False):
		self.agent.new_epoch(epoch_no)
		r = self._run_epoch(visible, trainingmode)
		if visible:
			self.visual.visualize_course_of_action(self.first_state, r[1], image_name="agent_path_" + str(epoch_no) + ".png")
		return r

	def run(self, epochs=1, visible=False, trainingmode=False):
		res = []
		for i in range(epochs):
			r = self.run_epoch(i, visible, trainingmode)
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
		return self.state[mid_n,mid_m] == self.target

	def get_current_state(self):
		"""return the current state """
		return self.state

	# is out of bounds
	def _is_oob(self):
		return np.sum(self.state) == 0

	def _execute_action(self, action):
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

	def get_best_possible_steps(self):
		x, y = np.where(self.state == self.target)
		mid_x = int(np.floor(self.grid_dims[0] / 2))
		mid_y = int(np.floor(self.grid_dims[1] / 2))
		return abs(mid_x - x[0]) + abs(mid_y - y[0])


class GaussSimulator(SimpleMatrixSimulator):
	""" uses heatmap with Gauss in the middle as environment """

	std = 3
	world_factor = 3 # the "world" has a size factor x grid_dims

	def _get_init_state(self,dims):
		"""initialize a random state that is a matrix with zeros and one one in it at a random position """
		(n,m) = dims
		N = self.world_factor * n
		M = self.world_factor * m
		mid_N = N//2
		mid_M = M//2
		# create world-state
		x = np.arange(0, N, 1, float)
		y = np.arange(0, M, 1, float)[:,np.newaxis]
		self.world_state = np.round( np.exp(-4*np.log(2) * ((x-mid_N)**2 + (y-mid_M)**2) / self.std**2) , 2)
		# generate state as window of the world-state
		# TODO: at the moment middle of gauss is always visible
		i = np.random.randint(mid_N-n+1,mid_N+1)
		j = np.random.randint(mid_M-m+1,mid_M+1)
		# avoid generation in the middle
		if i+n//2 == mid_N and j+m//2 == mid_M:
			i += np.random.choice([-1,1])
		self.i_world = i
		self.j_world = j
		return self.world_state[i:i+n,j:j+m]

	# is out of bounds (if at edge of world_state this is already oob)
	def _is_oob(self):
		(N,M) = self.world_state.shape
		(n,m) = self.grid_dims
		if self.i_world == 0 or self.j_world == 0 or self.i_world + n == N or self.j_world + m == M:
			return True
		else:
			return np.sum(self.state) == 0


import cv2


class ImageSimulator(SimpleMatrixSimulator):
	def __init__(self, agent, reward, img_path, grid_n, grid_m=1, orientation=0, max_steps=1000, visualizer=None,
				 bounded=True):
		super(ImageSimulator, self).__init__(agent, reward, grid_n, grid_m, orientation, max_steps, visualizer, bounded)
		self._load_and_preprocess_img(img_path)

	def _load_and_preprocess_img(self, path):
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur_factor = 5.
		blur_kernel = tuple([int(d / blur_factor + 1 if int(d / blur_factor) % 2 == 0 else 0) for d in self.grid_dims])
		img = cv2.GaussianBlur(img, blur_kernel, 10)
		world_dims = tuple([d * self.world_factor for d in self.grid_dims])
		img = cv2.resize(img, world_dims)
		# DEBUG, draw world dims frame around image
		view = img.copy()
		cv2.rectangle(view, (0, 0), (world_dims[0] - 1, world_dims[1] - 1), (255, 255, 255), 1)
		cv2.imwrite("tmp/view.png", view)
		# normalize image to [0, 1]
		img = np.array(img, np.float32)
		img_min = np.min(img)
		img_max = np.max(img)
		img -= img_min
		img /= img_max - img_min
		self.target = np.max(img)
		self.img = img

	def _get_init_state(self, dims):
		super(ImageSimulator, self)._get_init_state(dims)
		self.world_state = self.img
		state = self._extract_state_from_world(self.i_world, self.j_world)
		# DEBUG, draw current view
		# view = cv2.imread("tmp/view.png")
		# cv2.rectangle(view, (self.i_world, self.j_world), (self.i_world + dims[0], self.j_world + dims[1]),
		#			  (255, 255, 255), 1)
		#cv2.imwrite("tmp/view_curr.png", view)
		return state