from __future__ import division
from lat.Environment import Environment

from copy import copy
# from enum import Enum
import numpy as np
import time
import cv2


def enum(**enums):
	return type('Enum', (), enums)


Actions = enum(up=0, down=1, left=2, right=3)
Actions.all = [Actions.up, Actions.down, Actions.left, Actions.right]


class SimpleMatrixSimulator(Environment):
	world_factor = 3  # the "world" has a size factor x grid_dims
	window_gen_factor = 0.95  # decides where to sample init state from, 1 -> goal can be at the edge
	target = 1

	"""simulates an image frame environment for a learning agent"""

	def __init__(self, agent, reward, grid_n, grid_m=1, orientation=0, max_steps=1000, visualizer=None, bounded=True):
		self.agent = agent
		self.reward = reward
		self.visual = visualizer
		self.grid_dims = self._get_odd_dims(grid_n, grid_m)
		self.orientation = orientation
		self.max_steps = max_steps
		self.bounded = bounded

		self.state = None
		self.old_state = None
		self.all_states = None  # d x n x m dimensional numpy array, with d states of size n x m in it - used for visualization later
		# world state from which state is extracted
		self.world_state = None
		# top left corner of the window
		self.i_world = 0
		self.j_world = 0

	def _get_odd_dims(self, n, m):
		"""setting dimensions of image so that it has uneven number of elements on both dims to be able to center at the middle """
		if n % 2 == 0:
			n += 1
			print("Redefining dimension n to be ", n, "to have a middle pixel")
		if m % 2 == 0:
			m += 1
			print("Redefining dimension m to be ", m, "to have a middle pixel")
		return (n, m)

	def _get_init_state(self):
		""" sample an initial state (the top left corner of view-window to world-state) """
		(n, m) = self.grid_dims
		N = self.world_factor * n
		M = self.world_factor * m
		mid_N = N // 2
		mid_M = M // 2
		# create world-state
		self.world_state = np.zeros((N, M))
		self.world_state[mid_N, mid_M] = 1
		i = np.random.randint(mid_N - np.round(n * self.window_gen_factor, 0) + 1,
							  mid_N - np.round(n * (1 - self.window_gen_factor)) + 1)
		j = np.random.randint(mid_M - np.round(m * self.window_gen_factor, 0) + 1,
							  mid_M - np.round(m * (1 - self.window_gen_factor)) + 1)
		# avoid generation in the middle
		if i + n // 2 == mid_N and j + m // 2 == mid_M:
			i += np.random.choice([-1, 1])
		self.i_world = i
		self.j_world = j
		self.first_i = i
		self.first_j = j
		return self._extract_state_from_world(i, j)

	def _extract_state_from_world(self, i, j):
		(n, m) = self.grid_dims
		cut = self.world_state[i:i + n, j:j + m]
		return cut

	def _run_epoch(self, trainingmode=False):
		self.state = self._get_init_state()
		best = self.get_best_possible_steps()
		steps = []
		while len(steps) < self.max_steps:
			self.old_state = copy(self.state)
			action = self.agent.choose_action(self.state)
			steps.append(action)
			self.execute_action(action)
			is_oob = self._is_oob()
			if trainingmode:
				reward = self.reward.get_reward(self.old_state, self.state, is_oob)
				# print("reward is ",reward)
				self.agent.incorporate_reward(self.old_state, action, self.state, reward)
			if is_oob:
				return 0, steps, best
			elif self._at_goal(self.state):
				return 1, steps, best
		return 0, steps, best

	def run(self, epoch_no=1, visualize=False, trainingmode=False):
		self.agent.new_epoch(epoch_no)
		r = self._run_epoch(trainingmode)
		if visualize:
			timestamp = time.strftime("%Y%m%d_%H%M%S")
			image_name = timestamp + "_agent_path_epoch" + str(epoch_no).zfill(4)
			self.visual.visualize_course_of_action(self.world_state, self.first_i, self.first_j, self.grid_dims[0],
												   self.grid_dims[1], r[1], title="Path epoch {0}".format(epoch_no),
												   image_name=image_name)
		return r

	def _get_middle(self):
		""" returns middle of the state grid as tuple (mid_i, mid_j)"""
		(n, m) = self.grid_dims
		return (n // 2, m // 2)

	def _at_goal(self, state):
		""" returns True if at goal position and false otherwise """
		(mid_n, mid_m) = self._get_middle()
		return self.state[mid_n, mid_m] == self.target

	def get_current_state(self):
		"""return the current state """
		return self.state

	# is out of bounds (if at edge of world_state this is already oob)
	def _is_oob(self):
		(N, M) = self.world_state.shape
		(n, m) = self.grid_dims
		if self.i_world <= 0 or self.j_world <= 0 or self.i_world + n >= N or self.j_world + m >= M:
			return True
		else:
			return np.sum(self.state) == 0

	def execute_action(self, action):
		"""execute the action and therefore change the state """
		self.state = self._shift_image(action)

	def _shift_image(self, direction):
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
		xs, ys = np.where(self.state == self.target)
		mid_x = int(np.floor(self.grid_dims[0] / 2))
		mid_y = int(np.floor(self.grid_dims[1] / 2))
		calc_dist = lambda x, y: abs(mid_x - x) + abs(mid_y - y)
		dists = [calc_dist(x, y) for x, y in zip(xs, ys)]
		if len(dists) > 0:
			return np.min(dists)
		else:
			return -1


class GaussSimulator(SimpleMatrixSimulator):
	""" uses heatmap with Gauss in the middle as environment """

	std = 3
	world_factor = 3  # the "world" has a size factor x grid_dims

	def _get_init_state(self):
		"""initialize a random state that is a matrix with zeros and one one in it at a random position """
		(n, m) = self.grid_dims
		N = self.world_factor * n
		M = self.world_factor * m
		mid_N = N // 2
		mid_M = M // 2
		# create world-state
		x = np.arange(0, N, 1, float)
		y = np.arange(0, M, 1, float)[:, np.newaxis]
		self.world_state = np.round(np.exp(-4 * np.log(2) * ((x - mid_N) ** 2 + (y - mid_M) ** 2) / self.std ** 2), 2)
		# generate state as window of the world-state
		# TODO: at the moment middle of gauss is always visible
		i = np.random.randint(mid_N - n + 1, mid_N + 1)
		j = np.random.randint(mid_M - m + 1, mid_M + 1)
		# avoid generation in the middle
		if i + n // 2 == mid_N and j + m // 2 == mid_M:
			i += np.random.choice([-1, 1])
		self.i_world = i
		self.j_world = j
		return self._extract_state_from_world(i, j)


class ImageSimulator(SimpleMatrixSimulator):
	def __init__(self, agent, reward, img_path, grid_n, grid_m=1, orientation=0, max_steps=1000, visualizer=None,
				 bounded=True):
		super(ImageSimulator, self).__init__(agent, reward, grid_n, grid_m, orientation, max_steps, visualizer, bounded)
		self._load_and_preprocess_img(img_path)

	def _load_and_preprocess_img(self, path):
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# blur_factor = 1./10.
		# blur_kernel = tuple([int(d * blur_factor + 1 if int(d * blur_factor) % 2 == 0 else 0) for d in self.grid_dims])
		blur_kernel = (3, 3)
		img = cv2.GaussianBlur(img, blur_kernel, 0)
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
		img_diff = img_max - img_min
		if img_diff != 0:
			img -= img_min
			img /= img_max - img_min
		self.target = np.max(img)
		self.world_state = img

	def _get_init_state(self):
		""" sample an initial state (the top left corner of view-window to world-state) """
		(n, m) = self.grid_dims
		N = self.world_factor * n
		M = self.world_factor * m
		mid_N = N // 2
		mid_M = M // 2
		i = np.random.randint(mid_N - np.round(n * self.window_gen_factor, 0) + 1,
							  mid_N - np.round(n * (1 - self.window_gen_factor)) + 1)
		j = np.random.randint(mid_M - np.round(m * self.window_gen_factor, 0) + 1,
							  mid_M - np.round(m * (1 - self.window_gen_factor)) + 1)
		# avoid generation in the middle
		if i + n // 2 == mid_N and j + m // 2 == mid_M:
			i += np.random.choice([-1, 1])
		self.i_world = i
		self.j_world = j
		self.first_i = i
		self.first_j = j
		# DEBUG, draw current view
		view = cv2.imread("tmp/view.png")
		cv2.rectangle(view, (i, j), (i + self.grid_dims[0], j + self.grid_dims[1]), (255, 255, 255), 1)
		cv2.imwrite("tmp/view_curr.png", view)
		return self._extract_state_from_world(i, j)


class GrosserSternImageSimulator(ImageSimulator):
	""" world state is a gray scale google earth image of the grosser stern in berlin """
	world_factor = 20
	def _load_and_preprocess_img(self, path):
		""" preprocess the given image"""
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = img / float(np.max(img))  # renorm to 1 as max
		# perhaps rescale it if necessary
		world_dims = tuple([d * self.world_factor for d in self.grid_dims])
		img = cv2.resize(img, world_dims)
		self.img = img
		self.target = np.max(img)

	def _get_init_state(self):
		""" initial state is randomly sampled all over the picture except
		for when it sampled on the goal, then started again """
		(n, m) = self.grid_dims
		(N, M) = self.world_state.shape
		i = np.random.randint(np.round((1 - self.window_gen_factor) * N, 0), np.round(N * self.window_gen_factor - n, 0))
		j = np.random.randint(np.round((1 - self.window_gen_factor) * M, 0), np.round(M * self.window_gen_factor - m, 0))
		# avoid generation with goal reached, so redo sampling
		state = self._extract_state_from_world(i, j)
		self.i_world = i
		self.j_world = j
		self.first_i = i
		self.first_j = j
		if self._at_goal(state):
			state = self._get_init_state()
		return state

	def get_best_possible_steps(self):
		""" best possible steps is the distance in steps to the first 1 of gray scale image that appears within view """
		goal_locations = np.where(self.world_state == 1)
		goal_is = goal_locations[0]
		goal_js = goal_locations[1]
		(n, m) = self.grid_dims
		(i, j) = (self.i_world, self.j_world)
		distances = np.abs(goal_is - i - n/2) + np.abs(goal_js - j - m/2)
		return np.min(distances)

	def _at_goal(self, state):
		""" returns 1 if at goal or 0 if not (goal is to get a 1 into the middle of the view) """
		(n, m) = state.shape
		if (state[int(n/2), int(m/2)] == 1) > 0:
			return 1
		else:
			return 0

	def execute_action(self, action):
		world_state = self.world_state
		super(ImageSimulator, self).execute_action(action)
		i = self.i_world
		j = self.j_world
		(n, m) = self.grid_dims
		self.visual.visualize_current_state(world_state, i, j, n, m)

class ImageSimulatorSpecialSample(ImageSimulator):
	""" simulates image and samples with a special sampling (near center at beginning and farther away later) """

	epoch = None
	epoch_max = None
	special_sampling = False

	def use_special_sampling(self, epoch, max_epochs):
		""" switching special sampling mode on/off (True/False) """
		self.epoch_max = max_epochs
		self.epoch = epoch
		self.special_sampling = True

	def _get_init_state(self):
		if self.special_sampling:
			self._get_init_state_special(self.epoch_max, self.epoch)
			state = self._extract_state_from_world(self.i_world, self.j_world)
			return state
		else:
			return super(ImageSimulatorSpecialSample, self)._get_init_state()

	def _get_init_state_special(self, epochs, curr_epoch):
		""" sample an initial state (the top left corner of view-window to world-state), start with sampling  near
		center and begin to wander farther away"""
		(n, m) = self.grid_dims
		N = self.world_factor * n
		M = self.world_factor * m
		mid_N = N // 2
		mid_M = M // 2

		min_wgf = 0.55  # min 0.5 -> then sampled directly in the middle
		max_wgf = 1  # self.window_gen_factor
		win_gen_factor = min(max_wgf, min_wgf + (curr_epoch / (epochs * 0.6)) * (max_wgf - min_wgf))
		# print(win_gen_factor)
		lower_i = mid_N - np.round(n * win_gen_factor, 0) + 1
		upper_i = mid_N - np.round(n * (1 - win_gen_factor), 0) + 1

		lower_j = mid_M - np.round(m * win_gen_factor, 0) + 1
		upper_j = mid_M - np.round(m * (1 - win_gen_factor), 0) + 1

		i = np.random.randint(lower_i, upper_i)
		j = np.random.randint(lower_j, upper_j)

		# avoid generation in the middle
		if i + n // 2 == mid_N and j + m // 2 == mid_M:
			i += np.random.choice([-1, 1])
		self.i_world = i
		self.j_world = j
		self.first_i = i
		self.first_j = j
		return self.world_state[i:i + n, j:j + m]
