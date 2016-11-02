from random import random
import numpy as np
from scipy.ndimage.interpolation import shift
from lat import RobotAgent
from lat.Environment import Environment


class Simulator(Environment):
	NO_TARGET = 0
	TARGET = 1

	# agent is RobotAgent
	def __init__(self, agent, max_steps, grid_size):
		self.agent = agent
		self.max_steps = max_steps
		self.grid_size = grid_size
		self.state = None

	def rnd_pos(self):
		x = int(random() * self.grid_size)
		y = int(random() * self.grid_size)
		return x, y

	def rnd_pos_except_center(self):
		pos = self.rnd_pos()
		mid = np.floor(self.grid_size / 2)
		if pos == (mid, mid):
			return self.rnd_pos_except_center()
		else:
			return pos

	def get_init_state(self, target_pos):
		x, y = target_pos
		mat = np.full((self.grid_size, self.grid_size), self.NO_TARGET, np.int)
		mat[x, y] = self.TARGET
		return mat

	# is out of bounds
	def is_oob(self, state):
		# locate target
		pos = np.where(state == self.TARGET)
		# if target invisible then oob
		return len(pos[0]) == 0

	def get_middle(self):
		mid = int(np.floor(self.grid_size / 2))
		return mid, mid

	def is_success(self):
		x, y = self.get_middle()
		return self.state is not None and np.isclose(self.state[x, y], self.TARGET)

	def run(self, log=False):
		target_pos = self.rnd_pos_except_center()
		self.state = self.get_init_state(target_pos)
		steps = 0
		while steps < self.max_steps:
			old_state = self.state
			action = self.agent.choose_action(self.state)
			if action is None:
				return False
			# update state
			self.execute_action(action)
			success = self.is_success()
			reward = 1 if success else 0
			self.agent.incorporate_reward(old_state, action, self.state, reward)
			if success:
				return True
			if self.state is None:
				return False
		return False

	def get_current_state(self):
		return self.state

	def shift_state(self, action):
		if action == RobotAgent.Actions.up:
			return shift(self.state, [-1, 0], cval=0)
		elif action == RobotAgent.Actions.down:
			return shift(self.state, [1, 0], cval=0)
		elif action == RobotAgent.Actions.left:
			return shift(self.state, [0, -1], cval=0)
		else:
			return shift(self.state, [0, 1], cval=0)

	def execute_action(self, action):
		new_state = np.rint(self.shift_state(action)).astype(int)
		# only accept new state if not out of bounds
		self.state = new_state if not self.is_oob(new_state) else None
