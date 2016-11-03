from lat import Environment
import numpy as np

class Simulator(Environment):
	"""simulates an image frame environment for a learning agent"""
	def __init__(self, agent, reward, grid_n, grid_m=0, orientation=0, max_steps=1000):
		self.agent = agent
		self.reward = reward 
		self.grid_dims = self.get_proper_dims(grid_n,grid_m)
		self.orientation = orientation
		self.max_steps = max_steps

		self.state = None
		self.old_sate = None

	# setting dimensions of image so that it has uneven number of elements on both dims to be able to center at the middle
	def get_proper_dims(self,n,m):
		if n%2 != 0:
			n += 1
			print "Redefining dimension n to be ",n
		if m%2 != 0:
			m += 1
			print "Redefining dimension n to be ",m
		return (n,m)


	# initialize a random state that is a matrix with zeros and one one in it at a random position
	def get_rand_matrix_state(self,dims):
		state = np.zeros(dims)
		x = np.random.randint(0,dims[0])
		y = np.random.randint(0,dims[1])
		state[x,y] = 1
		return state

	# initialize the state to be a zero matrix with a gaussian distribution at a random position
	def get_rand_heat_state(self):
		pass

	def run(self, mode="test", visible=False):
		self.state = self.get_rand_matrix_state(self.grid_dims)

#TODO hier war ich eben
		steps = 0
		while steps < self.max_steps and not self.at_goal():
			
			action = self.agent.choose_action(self.state)
			self.execute_action(action)
			
			if self.agent.is_in_training_():
				#TODO get a reward and send it to the agent, which stores reward, action, state and new state 
				# to later use it for SGD, experience replay ,... just some way of training 
				pass

			if visible:
				self.stepwise_visualize()

	# returns True if at goal position and false otherwise
	def at_goal(self):
		current_i = np.argmax(self.state) / self.grid_dims[1]
		current_j = np.argmax(self.state) % self.grid_dims[1]
		goal_i = self.grid_dims(0)/2
		goal_j = self.grid_dims(1)/2
		if goal_i == current_i and goal_j == current_j:
			return True
		else
			return False

	# return the current state
	def get_current_state(self):
		return self.state

	# execute the action and therefore change the state
	def execute_action(self, action):
		pass

	# actual state change for the matrix image variant
	def shift_image(self,direction):
		pass

	# receive a reward from the reward object
	def calc_reward(self, state, old_state):
		return self.reward.get_reward(state, old_state)

	# just print matrix to stdout -1 old position of goal and 1 current position
	def stepwise_visualize(self):
		print self.state - self.old_state 