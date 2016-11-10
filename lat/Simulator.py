from lat.Environment import Environment

from copy import copy
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import os
import pylab 

class Actions(Enum):
	up = 0
	down = 1
	left = 2
	right = 3

	@staticmethod
	def all():
		return [a for a in Actions]


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
		self.all_states = None # d x n x m dimensional numpy array, with d states of size n x m in it - used for visualization later


	def get_proper_dims(self,n,m):
		"""setting dimensions of image so that it has uneven number of elements on both dims to be able to center at the middle """
		if n%2 == 0:
			n += 1
			print("Redefining dimension n to be ",n, "to have a middle pixel")
		if m%2 == 0:
			m += 1
			print("Redefining dimension m to be ",m, "to have a middle pixel")
		return (n,m)


	def get_rand_matrix_state(self,dims):
		"""initialize a random state that is a matrix with zeros and one one in it at a random position """
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


	def get_rand_heat_state(self):
		"""initialize the state to be a zero matrix with a gaussian distribution at a random position """
		pass


	def run(self, mode="test", visible=False, trainingmode=False):
		self.state = self.get_rand_matrix_state(self.grid_dims)
		if visible: 
			self.all_states = self.state[np.newaxis,:,:] # adds initial state to state list (and with that initializ state list)
		if visible: print(self.state)
		steps = 0
		while not self.lost_goal and steps < self.max_steps and not self.at_goal(self.state):
			steps += 1
			self.old_state = copy(self.state)
			action = self.agent.choose_action(self.state)
			self.execute_action(action)
			
			if trainingmode:
				#TODO get a reward and send it to the agent, which stores reward, action, state and new state 
				# to later use it for SGD, experience replay ,... just some way of training 
				print("reward is ",self.reward.get_reward(self.old_state, self.state, self.lost_goal))

			if visible:
				self.stepwise_visualize()
				self.all_states = np.concatenate( (self.all_states,self.state[np.newaxis,:,:]) , axis=0 ) # adds initial state to state list (and with that initializ state list)

	def visualize_path(self,agent_path_image_name="agent_path.png"):
		""" visualizes the collected states in a nice graphic """
		# agent_path = self.all_states.sum(axis=0)
		# agent_path_img = plt.imshow(agent_path,cmap="gray")
		# plt.savefig(agent_path_img)
		# print('Saved path png of agent to file ', os.path.abspath(__file__),'\\',agent_path_image_name)
		(cnt,n,m) = self.all_states.shape
		x = np.zeros(cnt)
		y = np.zeros(cnt)
		for i_state in range(cnt):
			(i,j) = self.get_goal_loc(self.all_states[i_state,:,:])
			x[i_state] = j
			y[i_state] = -i
		# plot path and start position as red o and final position as red x
		plt.plot(x,y,'b-',x[0],y[0],'ro',x[-1],y[-1],'rx')
		# set proper axis limits and remove ticks
		plt.xlim((-0.5,m-0.5))
		plt.ylim((-n+0.5,0.5))
		plt.xticks([])
		plt.yticks([])
		plt.savefig(agent_path_image_name)

	def get_goal_loc(self, state):
		"""report current location of the target where to focus on in the image / state """
		current_i = np.argmax(state) // self.grid_dims[1]
		current_j = np.argmax(state) % self.grid_dims[1]
		return (current_i, current_j)


	def at_goal(self, state):
		""" returns True if at goal position and false otherwise """
		(current_i, current_j) = self.get_goal_loc(state)
		goal_i = self.grid_dims[0]//2
		goal_j = self.grid_dims[1]//2
		if goal_i == current_i and goal_j == current_j:
			return True
		else:
			return False


	def get_current_state(self):
		"""return the current state """
		return self.state


	def execute_action(self, action):
		"""execute the action and therefore change the state """
		self.shift_image(action,self.bounded)


	def shift_image(self,direction,bounded=True):
		"""actual state change for the matrix image variant """

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


	def calc_reward(self, state, old_state):
		"""receive a reward from the reward object """
		return self.reward.get_reward(state, old_state)


	def stepwise_visualize(self):
		"""just print matrix to stdout -1 old position of goal and 1 current position """
		print(self.state)