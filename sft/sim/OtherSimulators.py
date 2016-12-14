class PathSimExpansiveSampler(PathSimulator):
	""" simple path simulation -> just a line with target on it + expansive sampling """
	# IMPORTANT IF NAME IS CHANGED ALSO LOOK INTO 'RUN' OF EVALUATOR TO CHANGE NAME

	epochs = None  # contains the overall max of epochs
	curr_epoch = 1  # contains current number of epochs

	pct_epochs_until_max = 0.5  # after what percentage of epochs to reach max sampling distance
	pct_min_dist_in_grid_size = 0.4  # at which distance to start sampling (in percent of grid-size)

	def _get_init_state(self):
		""" use special sampling  """
		self.state = None
		# we need to generate a new world for every run
		self._initialize_world()
		# here special expansive sampling happens
		# get parameters
		i, j = self._get_target_loc()
		N, M = self.world_image.shape
		n, m = self.grid_dims
		# get limits
		i_lim, j_lim = self._get_limits(i, j, N, M, n, m)
		# get start position of upper left corner of view
		i_0 = np.random.randint(i_lim[0], i_lim[1] + 1)
		j_0 = np.random.randint(j_lim[0], j_lim[1] + 1)
		# set state, start-state and extract the view
		self.i_world, self.j_world = i_0, j_0
		self.first_i, self.first_j = i_0, j_0
		self.state = self._extract_state_from_world(i_0, j_0)
		self.curr_epoch += 1
		return self.state

	def _get_limits(self, i_goal, j_goal, n_world, m_world, n_grid, m_grid):
		i_lim = self._get_limit(i_goal, n_grid, n_world)
		j_lim = self._get_limit(j_goal, m_grid, m_world)
		return i_lim, j_lim

	def _get_limit(self, i, n, N):
		e = self.curr_epoch
		e_max_reached = self.epochs * self.pct_epochs_until_max
		if e >= e_max_reached:
			i_lim = (2 * n, N - 3 * n)
		else:
			min_dist = n * self.pct_min_dist_in_grid_size
			max_dist = N
			dist = np.abs(int((max_dist - min_dist - i - 2 * n) * e / e_max_reached))
			i_low = i - min_dist - dist - int(n / 2)
			i_high = i + min_dist + dist - int(n / 2)
			i_lim = (i_low, i_high)
			i_lim = self._correct_to_within_world_state(i_lim, n, N)
		return i_lim

	def _correct_to_within_world_state(self, i_lim, n, N):
		i_low = i_lim[0]
		i_high = i_lim[1]
		if i_low - 2 * n < 0:
			i_low = 2 * n
		if i_high > N - 3 * n:
			i_high = N - 3 * n
		if i_low > i_high:
			i_high = i_low + 1
		return i_low, i_high

	def _get_target_loc(self):
		# TODO: there must be a better way to extract the goal location (middle of circle)
		ij_target = np.where(self.world_image == 1)
		ij_target = np.mean(ij_target, axis=1).astype(int)
		assert ij_target.size == 2, "target location in expansive sampler does not contain 2 indices"
		return ij_target

	# somehow needs to know EPOCHS and current epoch

	def restartExpansiveSampling(self, epochs):
		self.epochs = epochs
		self.curr_epoch = 1

	# is out of bounds (if at edge of world_state this is already oob)
	def _is_oob(self):
		(N, M) = self.world_image.shape
		(n, m) = self.grid_dims
		if self.i_world <= 0 or self.j_world <= 0 or self.i_world + n >= N or self.j_world + m >= M:
			return True
		else:
			return False


class PathSimExpansiveSamplerOnPath(PathSimExpansiveSampler):
	def _get_init_state(self):
		state = super(PathSimExpansiveSamplerOnPath, self)._get_init_state()
		# generate new states as long as there is no road visible or the target is visible
		while is_state_empty(state):
			state = super(PathSimExpansiveSamplerOnPath, self)._get_init_state()
		return state


class PathSimulatorSimple(PathSimulator):
	def _initialize_world(self, path_length=-1):
		# always generate straight line path with length
		length = 1
		super(PathSimulatorSimple, self)._initialize_world(length)

	def sample_step_from(self, bbox, prev_node, step_size_min=-1):
		# step size is minimum five views
		step_size_min = min(self.view_size.tuple()) * 5
		return super(PathSimulatorSimple, self).sample_step_from(bbox, prev_node, step_size_min)


class PathSimulatorSimpleOnPath(PathSimulatorSimple):
	def _get_init_state(self):
		state = super(PathSimulatorSimpleOnPath, self)._get_init_state()
		# generate new states as long as there is no road visible or the target is visible
		while is_state_empty(state) or contains_target(state, 1):
			state = super(PathSimulatorSimpleOnPath, self)._get_init_state()
		return state


class PathSimSimpleExpansiveSampler(PathSimulatorSimple):
	""" simple path simulation -> just a line with target on it + expansive sampling """
	# IMPORTANT IF NAME IS CHANGED ALSO LOOK INTO 'RUN' OF EVALUATOR TO CHANGE NAME

	epochs = None  # contains the overall max of epochs
	curr_epoch = 1  # contains current number of epochs

	pct_epochs_until_max = 0.5  # after what percentage of epochs to reach max sampling distance
	pct_min_dist_in_grid_size = 0.4  # at which distance to start sampling (in percent of grid-size)

	def _get_init_state(self):
		""" use special sampling  """
		self.state = None
		# we need to generate a new world for every run
		self._initialize_world()
		# here special expansive sampling happens
		# get parameters
		i, j = self._get_target_loc()
		N, M = self.world_image.shape
		n, m = self.grid_dims
		# get limits
		i_lim, j_lim = self._get_limits(i, j, N, M, n, m)
		# get start position of upper left corner of view
		i_0 = np.random.randint(i_lim[0], i_lim[1] + 1)
		j_0 = np.random.randint(j_lim[0], j_lim[1] + 1)
		# set state, start-state and extract the view
		self.i_world, self.j_world = i_0, j_0
		self.first_i, self.first_j = i_0, j_0
		self.state = self._extract_state_from_world(i_0, j_0)
		self.curr_epoch += 1
		return self.state

	def _get_limits(self, i_goal, j_goal, n_world, m_world, n_grid, m_grid):
		i_lim = self._get_limit(i_goal, n_grid, n_world)
		j_lim = self._get_limit(j_goal, m_grid, m_world)
		return i_lim, j_lim

	def _get_limit(self, i, n, N):
		e = self.curr_epoch
		e_max_reached = self.epochs * self.pct_epochs_until_max
		if e >= e_max_reached:
			i_lim = (2 * n, N - 3 * n)
		else:
			min_dist = n * self.pct_min_dist_in_grid_size
			max_dist = N
			dist = np.abs(int((max_dist - min_dist - i - 2 * n) * e / e_max_reached))
			i_low = i - min_dist - dist - int(n / 2)
			i_high = i + min_dist + dist - int(n / 2)
			i_lim = (i_low, i_high)
			i_lim = self._correct_to_within_world_state(i_lim, n, N)
		return i_lim

	def _correct_to_within_world_state(self, i_lim, n, N):
		i_low = i_lim[0]
		i_high = i_lim[1]
		if i_low - 2 * n < 0:
			i_low = 2 * n
		if i_high > N - 3 * n:
			i_high = N - 3 * n
		if i_low > i_high:
			i_high = i_low + 1
		return i_low, i_high

	def _get_target_loc(self):
		# TODO: there must be a better way to extract the goal location (middle of circle)
		ij_target = np.where(self.world_image == 1)
		ij_target = np.mean(ij_target, axis=1).astype(int)
		assert ij_target.size == 2, "target location in expansive sampler does not contain 2 indices"
		return ij_target

	# somehow needs to know EPOCHS and current epoch

	def restartExpansiveSampling(self, epochs):
		self.epochs = epochs
		self.curr_epoch = 1

	# is out of bounds (if at edge of world_state this is already oob)
	def _is_oob(self):
		(N, M) = self.world_image.shape
		(n, m) = self.grid_dims
		if self.i_world <= 0 or self.j_world <= 0 or self.i_world + n >= N or self.j_world + m >= M:
			return True
		else:
			return False


class PathSimSimpleExpansiveSamplerOnPath(PathSimSimpleExpansiveSampler):
	def _get_init_state(self):
		state = super(PathSimSimpleExpansiveSamplerOnPath, self)._get_init_state()
		# generate new states as long as there is no road visible or the target is visible
		while is_state_empty(state):
			state = super(PathSimSimpleExpansiveSamplerOnPath, self)._get_init_state()
		return state


class PathSimExpSplImages(PathSimSimpleExpansiveSampler):
	world_images = []  # array used to store the already created images
	world_img_path = "tmp/line-worldstates"
	world_img_format = ".png"

	def _get_init_state(self):
		""" use special sampling and already created images """
		self.state = None
		# we need to generate a new world for every run <- sampled from existing line configurations
		self.world_state = self._sample_from_images(only_hor_vert=True, only_end=True)
		# here special expansive sampling happens
		# get parameters
		i, j = self._get_target_loc()
		N, M = self.world_state.shape
		n, m = self.grid_dims
		# get limits
		i_lim, j_lim = self._get_limits(i, j, N, M, n, m)
		# get start position of upper left corner of view
		i_0 = np.random.randint(i_lim[0], i_lim[1] + 1)
		j_0 = np.random.randint(j_lim[0], j_lim[1] + 1)
		# set state, start-state and extract the view
		self.i_world, self.j_world = i_0, j_0
		self.first_i, self.first_j = i_0, j_0
		self.state = self._extract_state_from_world(i_0, j_0)
		self.curr_epoch += 1
		return self.state

	def _sample_from_images(self, only_hor_vert=True, only_end=True):
		if len(self.world_images) == 0:
			self._load_images(only_hor_vert, only_end=True)
		i_images = range(len(self.world_images))
		return self.world_images[np.random.choice(i_images)]

	def _load_images(self, only_hor_vert=True, only_end=True):
		dir_list = os.listdir(self.world_img_path)
		for f in dir_list:
			if f.endswith(self.world_img_format):
				f = self.world_img_path + "/" + f
				img = cv2.imread(f)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img = img / float(np.max(img))  # renorm to 1 as max
				self.world_images.append(img)


class PathSimExpSplImagesOnPath(PathSimExpSplImages):
	def _get_init_state(self):
		state = super(PathSimExpSplImagesOnPath, self)._get_init_state()
		# generate new states as long as there is no road visible or the target is visible
		while is_state_empty(state):
			state = super(PathSimExpSplImagesOnPath, self)._get_init_state()
		return state


def is_state_empty(state):
	return np.sum(state) == 0


def contains_target(state, target):
	return np.where(state == target)[0].size > 0