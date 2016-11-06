from Reward import Reward
import numpy as np


# simplest version, returning 10 if at goal, -10 if goal lost from focus, 0 otherwise
class RewardAtTheEnd(Reward):

	def get_reward(self, old_state, new_state, lost=False):
		if lost:
			return -10
		elif self.at_goal(new_state):
			return 10
		else:
			return 0


	# report current location of the target where to focus on in the image / state
	def get_goal_loc(self, state):
		current_i = np.argmax(state) // state.shape[1]
		current_j = np.argmax(state) % state.shape[1]
		return (current_i, current_j)


	# returns True if at goal position and false otherwise
	def at_goal(self, state):
		(current_i, current_j) = self.get_goal_loc(state)
		goal_i = state.shape[0]//2
		goal_j = state.shape[1]//2
		if goal_i == current_i and goal_j == current_j:
			return True
		else:
			return False



# linear version, returning 10 if at goal, -10 if goal lost from focus, 1 if decreasing distane to goal, -1 if distance increases
class LinearReward(RewardAtTheEnd):

	def get_reward(self, old_state, new_state, lost=False):
		if lost:
			return -10
		elif self.at_goal(new_state):
			return 10
		elif self.distance_decreases(old_state,new_state):
			return 1
		else:
			return -1


	# calculates distance to goal for old and new state and returns True if it decreased and False otherwise
	def distance_decreases(self,old_state,new_state):
		(i_o,j_o) = self.get_goal_loc(old_state)
		(i_n,j_n) = self.get_goal_loc(new_state)
		(n,m) = new_state.shape
		return np.abs(i_o - n//2) + np.abs(j_o - m//2) > np.abs(i_n - n//2) + np.abs(j_n - m//2)