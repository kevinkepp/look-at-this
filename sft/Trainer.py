import os
import pkgutil
from importlib import import_module

import numpy as np
import threading

from sft.Actions import Actions
from sft.State import State
from sim.Simulator import Simulator


class Trainer(object):
	WORLD_CONFIG_NAME = "world"
	AGENT_CONFIG_NAME_DIR = "agents"

	def run(self, experiment, threaded=True):
		world_config, agent_configs = self.get_configs(experiment)
		scenarios = self.init_scenarios(world_config)
		for agent in agent_configs:
			if threaded:
				thread = threading.Thread(target=self.run_agent, args=(agent, scenarios))
				thread.daemon = False
				thread.start()
			else:
				self.run_agent(agent, scenarios)

	def get_configs(self, experiment):
		world_config = import_module("." + self.WORLD_CONFIG_NAME, experiment.__name__)
		agent_configs = []
		experiment_dir = os.path.dirname(experiment.__file__)
		agents_dir = os.path.join(experiment_dir, self.AGENT_CONFIG_NAME_DIR)
		for loader, module, is_pkg in pkgutil.iter_modules([agents_dir]):
			if not is_pkg and module != "__init__":
				agent_config = import_module("." + module, experiment.__name__ + "." + self.AGENT_CONFIG_NAME_DIR)
				agent_configs.append(agent_config)
		return world_config, agent_configs

	def init_scenarios(self, config):
		return [config.world_gen.get_next() for n in range(config.epochs)]

	def run_agent(self, config, scenarios):
		for n in range(config.epochs):  # TODO check whether epochs in config
			success, action_hist = self.run_epoch(config, n, scenarios[n])
			# TODO log success and action_hist
			# if n % (epochs / 100) == 0:
			print("Agent {0} - Epoch {1}: {2}".format(config.__name__, n, success))

	def run_epoch(self, config, epoch, scenario):
		sim = Simulator(config.view_size)
		sim.initialize(scenario.world, scenario.pos)
		agent = config.agent
		reward = config.reward
		eps = config.epsilon_update.get_value(epoch)
		action_hist = []
		while len(action_hist) < config.max_steps:
			view = sim.get_current_view()
			state = self.get_state(config, view, action_hist)
			action = agent.choose_action(state, eps)
			action_hist.append(action)
			view2 = sim.update_view(action)
			state2 = self.get_state(config, view2, action_hist) if view2 is not None else None
			reward_value = reward.get_reward(view, view2)
			agent.incorporate_reward(state, action, state2, reward_value)
			if sim.is_oob():
				return 0, action_hist
			elif sim.is_at_target():
				return 1, action_hist
		return 0, action_hist

	def get_state(self, config, view, action_hist):
		actions = np.zeros([config.action_hist_len, len(Actions.all)])
		# take last n actions, this will be smaller or empty if there are not enough actions
		last_actions = action_hist[-config.action_hist_len:]
		for i in range(len(last_actions)):
			action = last_actions[i]
			actions[i] = Actions.get_one_hot(action)
		return State(view, actions)
