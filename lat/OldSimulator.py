from __future__ import division
from random import random
import numpy as np
from scipy.ndimage.interpolation import shift
from lat.Environment import Environment


def enum(**enums):
    return type('Enum', (), enums)

Actions = enum(up=0, down=1, left=2, right=3)
Actions.all = [Actions.up, Actions.down, Actions.left, Actions.right]


class Simulator(Environment):
	NO_TARGET = 0
	TARGET = 1

	# agent is RobotAgent
	def __init__(self, agent, reward, grid_size_n, grid_size_m, max_steps=100, bounded=False):
		self.agent = agent
		self.reward = reward
		if grid_size_n != grid_size_m:
			raise ValueError("Simulator only supports square grids")
		self.grid_size = grid_size_n
		self.max_steps = max_steps
		self.state = None

	def _rnd_pos(self):
		x = int(random() * self.grid_size)
		y = int(random() * self.grid_size)
		return x, y

	def _rnd_pos_except_center(self):
		pos = self._rnd_pos()
		mid = np.floor(self.grid_size / 2)
		if pos == (mid, mid):
			return self._rnd_pos_except_center()
		else:
			return pos

	def _get_init_state(self, target_pos):
		x, y = target_pos
		mat = np.full((self.grid_size, self.grid_size), self.NO_TARGET, np.int)
		mat[x, y] = self.TARGET
		return mat

	# is out of bounds
	def _is_oob(self):
		return np.sum(self.state) == 0

	def _get_middle(self):
		mid = int(np.floor(self.grid_size / 2))
		return mid, mid

	def _is_success(self):
		x, y = self._get_middle()
		return self.state[x, y] == self.TARGET

	def _get_best_possible_steps(self):
		x, y = np.where(self.state == self.TARGET)
		mid = int(np.floor(self.grid_size / 2))
		return abs(mid - x[0]) + abs(mid - y[0])

	def _run_epoch(self, trainingmode):
		target_pos = self._rnd_pos_except_center()
		self.state = self._get_init_state(target_pos)
		best = self._get_best_possible_steps()
		steps = []
		while len(steps) < self.max_steps:
			action = self.agent.choose_action(self.state)
			if action is None:
				return 0, steps, best
			steps.append(action)
			# update state
			old_state = self.state
			self.execute_action(action)
			# check if success or out of bounds (failure)
			success = self._is_success()
			oob = self._is_oob()
			# calculate and incorporate reward if in trainingmode
			if trainingmode:
				terminal = success or oob
				reward = self.reward.get_reward(old_state, self.state)
				new_state = self.state if not terminal else None
				self.agent.incorporate_reward(old_state, action, new_state, reward)
			if success:
				return 1, steps, best
			if oob:
				return 0, steps, best
		return 0, steps, best

	def run(self, epochs=1, trainingmode=False):
		res = []
		print_steps = 10
		for i in range(epochs):
			self.agent.new_epoch(i)
			r = self._run_epoch(trainingmode)
			if i % int(epochs / print_steps) == 0:
				print("Epoch {0}/{1}".format(i, epochs))
			res.append(r)
		return res

	def get_current_state(self):
		return self.state

	def _shift_state(self, action):
		if action == Actions.up:
			return shift(self.state, [-1, 0], cval=0)
		elif action == Actions.down:
			return shift(self.state, [1, 0], cval=0)
		elif action == Actions.left:
			return shift(self.state, [0, -1], cval=0)
		else:
			return shift(self.state, [0, 1], cval=0)

	def execute_action(self, action):
		self.state = np.rint(self._shift_state(action)).astype(int)