from random import random
import numpy as np
from lat import RobotAgent
from lat.Environment import Environment


class Simulator(Environment):
	VIEW_SIZE = 3  # 3x3

	# agent is RobotAgent
	def __init__(self, agent, max_steps):
		self.agent = agent
		self.max_steps = max_steps

	def rnd_pos(self):
		x = int(random() * self.VIEW_SIZE)
		y = int(random() * self.VIEW_SIZE)
		return x, y

	def rnd_pos_except_center(self):
		pos = self.rnd_pos()
		mid = np.floor(self.VIEW_SIZE/2)
		if pos == (mid, mid):
			return self.rnd_pos_except_center()
		else:
			return pos

	def get_init_state(self, goal_pos):
		x, y = goal_pos
		mat = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE))
		mat[x, y] = 1
		return mat

	def move(self, action, old_state):
		x, y = old_state
		if action == RobotAgent.Actions.up:
			return x, y + 1
		elif action == RobotAgent.Actions.down:
			return x, y - 1
		elif action == RobotAgent.Actions.left:
			return x - 1, y
		else:
			return x + 1, y

	def is_oob(self, state):
		x, y = state
		max_v = np.floor(self.VIEW_SIZE / 2)
		return np.abs(x) > max_v or np.abs(y) > max_v

	def train_once(self):
		goal_pos = self.rnd_pos()
		state = self.get_init_state(goal_pos)
		steps = 0
		# don't place on goal state
		while state == self.goal:
			state = self.rnd_state()
		while steps < self.max_steps:
			action = self.agent.choose_action(state)
			new_state = self.move(action, state)
			# check out of bounds
			oob = self.is_oob(new_state)
			new_state = None if oob else new_state
			success = new_state == self.goal
			reward = 1 if success else 0
			self.agent.incorporate_reward(state, action, new_state, reward)
			if success:
				return True
			if oob:
				return False
		return False

	# epochs: no of training epochs
	def train(self, epochs):
		print("Training " + str(epochs) + " epochs")
		for i in range(epochs):
			print("Epoch " + str(i) + ": " + str(self.train_once()))
