from Environment import Environment
from RobotAgent import Actions
from copy import copy

import numpy as np


class SimpleMatrixSimulator(Environment):
	"""simulates an image frame environment for a learning agent"""
	def __init__(self, agent, reward, grid_n, grid_m=1, orientation=0, max_steps=1000, bounded=True):
		self.agent = agent
		self.reward = reward 
		self.grid_dims = self.get_proper_dims(grid_n,grid_m)
		self.orientation = orientation
		self.max_steps = max_steps
		self.bounded = bounded

		self.state = None
		self.old_state = None
		self.lost_goal = False


	# setting dimensions of image so that it has uneven number of elements on both dims to be able to center at the middle
	def get_proper_dims(self,n,m):
		if n%2 == 0:
			n += 1
			print("Redefining dimension n to be ",n, "to have a middle pixel")
		if m%2 == 0:
			m += 1
			print("Redefining dimension m to be ",m, "to have a middle pixel")
		return (n,m)


	# initialize a random state that is a matrix with zeros and one one in it at a random position
	def get_rand_matrix_state(self,dims):
		state = np.zeros(dims)
		x = np.random.randint(0,dims[0])
		y = np.random.randint(0,dims[1])
		# avoid generation in the center
		if  dims[0] != 1 	and x == dims[0]//2:
			x += np.random.choice([-1,1],1)[0]
		if dims[1] != 1 	and y == dims[1]//2:
			y += np.random.choice([-1,1],1)[0]
		state[x,y] = 1
		return state


	# initialize the state to be a zero matrix with a gaussian distribution at a random position
	def get_rand_heat_state(self):
		pass


	def run(self, mode="test", visible=False):
		self.state = self.get_rand_matrix_state(self.grid_dims)
		if visible: print(self.state)
		is_training = self.agent.is_in_training_mode()
		steps = 0
		while not self.lost_goal and steps < self.max_steps and not self.at_goal(self.state):
			steps += 1
			self.old_state = copy(self.state)
			action = self.agent.choose_action(self.state)
			self.execute_action(action)
			
			if is_training:
				#TODO get a reward and send it to the agent, which stores reward, action, state and new state 
				# to later use it for SGD, experience replay ,... just some way of training 
				print("reward is ",self.reward.get_reward(self.old_state, self.state, self.lost_goal))

			if visible:
				self.stepwise_visualize()


	# report current location of the target where to focus on in the image / state
	def get_goal_loc(self, state):
		current_i = np.argmax(state) // self.grid_dims[1]
		current_j = np.argmax(state) % self.grid_dims[1]
		return (current_i, current_j)


	# returns True if at goal position and false otherwise
	def at_goal(self, state):
		(current_i, current_j) = self.get_goal_loc(state)
		goal_i = self.grid_dims[0]//2
		goal_j = self.grid_dims[1]//2
		if goal_i == current_i and goal_j == current_j:
			return True
		else:
			return False


	# return the current state
	def get_current_state(self):
		return self.state


	# execute the action and therefore change the state
	def execute_action(self, action):
		self.shift_image(action,self.bounded)


	# actual state change for the matrix image variant
	def shift_image(self,direction,bounded=True):

		if Actions.up == direction:
			(i,j) = self.get_goal_loc(self.state)
			self.state[i,j] = 0
			i += 1
			if i > self.grid_dims[0]-1: # target at bottom edge, up would move target out of frame
				if bounded:
					i -=1
					self.state[i,j] = 1
				else:
					self.lost_goal = True
			else: 
				self.state[i,j] = 1

		elif Actions.right == direction:
			(i,j) = self.get_goal_loc(self.state)
			self.state[i,j] = 0
			j -= 1
			if j < 0: # target at left edge, right would move target out of frame
				if bounded:
					j +=1
					self.state[i,j] = 1
				else:
					self.lost_goal = True
			else: 
				self.state[i,j] = 1


		elif Actions.down == direction:
			(i,j) = self.get_goal_loc(self.state)
			self.state[i,j] = 0
			i -= 1
			if i < 0: # target at top edge, down would move target out of frame
				if bounded:
					i += 1
					self.state[i,j] = 1
				else:
					self.lost_goal = True
			else: 
				self.state[i,j] = 1

		elif Actions.left == direction:
			(i,j) = self.get_goal_loc(self.state)
			self.state[i,j] = 0
			j +=1
			if j > self.grid_dims[1]-1: # target at right edge, left would move target out of picture
				if bounded:
					j -=1
					self.state[i,j] = 1
				else:
					self.lost_goal = True
			else: 
				self.state[i,j] = 1


	# receive a reward from the reward object
	def calc_reward(self, state, old_state):
		return self.reward.get_reward(state, old_state)


	# just print matrix to stdout -1 old position of goal and 1 current position
	def stepwise_visualize(self):
		print(self.state)