import numpy as np

from sft.Actions import Actions
from sft.State import State


class Trainer:
	def __init__(self, config):
		self.config = config

	def run(self):
		epochs = self.config.epochs
		sim = self.config.simulator
		agent = self.config.agent
		reward = self.config.reward
		eps_update = self.config.epsilon_update
		for n in range(epochs):
			eps = eps_update.get_value(n)
			success, action_hist = self.run_epoch(sim, agent, reward, eps)
			# TODO log success and action_hist
			# if n % (epochs / 100) == 0:
			print("Epoch {0}: {1}".format(n, success))
			sim.reset()

	def run_epoch(self, sim, agent, reward, eps):
		action_hist = []
		while len(action_hist) < self.config.max_steps:
			view = sim.get_current_view()
			state = self.get_state(view, action_hist)
			action = agent.choose_action(state, eps)
			action_hist += [action]
			view2 = sim.update_view(action)
			state2 = self.get_state(view2, action_hist)
			reward_value = reward.get_reward(view, view2)
			agent.incorporate_reward(state, action, state2, reward_value)
			if sim.is_oob():
				return 0, action_hist
			elif sim.is_at_target():
				return 1, action_hist
		return 0, action_hist

	def get_state(self, view, action_hist):
		actions = np.zeros([self.config.action_hist_len, len(Actions.all)])
		# take last n actions, this will be smaller or empty if there are not enough actions
		last_actions = action_hist[-self.config.action_hist_len:]
		for i in range(len(last_actions)):
			action = last_actions[i]
			one_hot = Actions.get_one_hot(action)
			actions[i] = one_hot
		return State(view, actions)
