from lat.RobotAgent import RobotAgent


class QAgent(RobotAgent):
	# experience
	exp = {}

	# alpha: learning rate
	# gamma: discount factor
	# q_init: initializer function for q values
	def __init__(self, actions, alpha, gamma, q_init):
		self._actions = actions
		self._alpha = alpha
		self._gamma = gamma
		self._q_init = q_init

	def q_values(self, state):
		state_str = str(state)
		if state_str not in self.exp:
			self.exp[state_str] = {}
		for action in self._actions:
			if action not in self.exp[state_str]:
				self.exp[state_str][action] = self._q_init(state, action)
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
		# get index of maximum q value for current state
		return self.max_q_value(curr_state)[0]

	def incorporate_reward(self, old_state, action, new_state, reward):
		q_old = self.q_value(old_state, action)
		# get max q value of new state
		q_max = self.max_q_value(new_state)[1]
		# update experience
		q_old += self._alpha * (reward + self._gamma * q_max - q_old)
		self.update_q(old_state, action, q_old)

	def new_epoch(self):
		pass
