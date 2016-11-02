from lat.RobotAgent import RobotAgent, Actions


class QLearningAgent(RobotAgent):
	# experience
	exp = {}

	# alpha: learning rate
	# gamma: discount factor
	# q_init: initializer function for q values
	def __init__(self, alpha, gamma, q_init):
		self.alpha = alpha
		self.gamma = gamma
		self.q_init = q_init

	def applicable_actions(self, curr_state):
		# applicable actions are all robot actions (for now)
		return Actions

	def q_values(self, state):
		state_str = str(state)
		if state_str not in self.exp:
			self.exp[state_str] = {}
		for action in self.applicable_actions(state):
			if action not in self.exp[state_str]:
				self.exp[state_str][action] = self.q_init(state, action)
		return self.exp[state_str]

	def q_value(self, state, action):
		return self.q_values(state)[action]

	def max_q_value(self, state):
		qs = self.q_values(state)
		q_max = max(qs, key=qs.get)
		return q_max, qs[q_max]

	def update_q(self, old_state, action, value):
		state_str = str(old_state)
		self.exp[state_str][action] = value

	def choose_action(self, curr_state):
		return self.max_q_value(curr_state)[0]

	def incorporate_reward(self, old_state, action, new_state, reward):
		q_old = self.q_value(old_state, action)
		# get max value of resulting state
		q_max = self.max_q_value(new_state)[1] if new_state is not None else 0.
		# update experience
		q_old += self.alpha * (reward + self.gamma * q_max - q_old)
		self.update_q(old_state, action, q_old)
