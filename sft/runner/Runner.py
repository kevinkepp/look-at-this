import os
import pkgutil
from importlib import import_module

from abc import abstractmethod

import threading
import time
import random
import sys

from sft.State import State
from sft.sim.Simulator import Simulator


class Runner(object):
	WORLD_CONFIG_NAME = "world"
	AGENT_CONFIG_NAME_DIR = "agents"

	def get_configs(self, experiment):
		world_config = self.get_world_config(experiment)
		agent_configs = self.get_agent_configs(experiment)
		return world_config, agent_configs

	def get_world_config(self, experiment):
		return import_module("." + self.WORLD_CONFIG_NAME, experiment.__name__)

	def get_agent_configs(self, experiment):
		agent_configs = []
		experiment_dir = os.path.dirname(experiment.__file__)
		agents_dir = os.path.join(experiment_dir, self.AGENT_CONFIG_NAME_DIR)
		for loader, module, is_pkg in pkgutil.iter_modules([agents_dir]):
			if not is_pkg and module != "__init__":
				agent_config = import_module("." + module, experiment.__name__ + "." + self.AGENT_CONFIG_NAME_DIR)
				agent_configs.append(agent_config)
		return agent_configs

	def run_agent(self, config, scenarios, seed=None):
		logger = config.logger
		if seed is not None:
			self.set_seed(seed)
			logger.log_message("Using seed %s for running agent" % str(seed))
		logger.log_message("{0} - Start running {1} episodes".format(config.__name__, config.epochs))
		sim = Simulator(config.view_size)
		time_start = time.time()
		for n in range(len(scenarios)):
			scenario = scenarios[n]
			success, actions = self.run_epoch(config, sim, n, scenario)
			logger.log_results(actions, success)
			logger.log_message("{0} - Episode {1} - Success: {2} - Steps: {3}".format(config.__name__, n, success, len(actions)))
			if n % config.model_persist_steps == 0:
				logger.log_model(config.model, str(n))
			logger.next_epoch()
			config.agent.new_episode()
		time_diff = time.time() - time_start
		logger.log_model(config.model, str(len(scenarios) - 1))
		logger.log_message("{0} - Finished training, took {1} seconds".format(config.__name__, time_diff))
		logger.close_files()

	def run_epoch(self, config, sim, epoch, scenario):
		"""Run training episode with given initial scenario"""
		sim.initialize(scenario.world, scenario.pos)
		eps = self._get_eps(config, epoch)
		config.logger.log_parameter("epsilon", eps)
		actions = []
		while len(actions) < config.max_steps:
			# result is 1 for on target, 0 for oob and -1 for non-terminal
			res = self.run_step(actions, sim, config.agent, config.reward, eps)
			if res != -1:
				return res, actions
		# if max steps are exceeded episode was not successful
		return 0, actions

	def run_step(self, past_actions, sim, agent, reward, eps):
		"""Conducts training step and returns 1 for success, 0 for loss and -1 for non-terminal"""
		view = sim.get_current_view()
		state = self.get_state(view, past_actions)
		action = agent.choose_action(state, eps)
		past_actions.append(action)
		view2 = sim.update_view(action)
		reward_value = reward.get_reward(view, view2)
		oob = sim.is_oob()
		at_target = sim.is_at_target()
		# epoch ends when agent runs out of bounds or hits the target
		terminal = oob or at_target
		# if new state is terminal None will be given to agent
		state2 = self.get_state(view2, past_actions) if not terminal else None
		self._incorp_agent_reward(agent, state, action, state2, reward_value)
		if at_target:
			return 1
		elif oob:
			return 0
		else:
			return -1

	def get_state(self, view, all_actions):
		return State(view, all_actions)

	def set_seed(self, seed=None):
		if seed is None:
			seed = self.gen_seed()
		random.seed(seed)
		return seed

	def gen_seed(self):
		return random.randint(0, sys.maxint)

	def gen_seeds(self, nb_seeds):
		return [self.gen_seed() for i in range(nb_seeds)]

	"""later used by Trainer to run epsilon update and by Tester to set epsilon to wished value"""
	@abstractmethod
	def _get_eps(self, config, epoch):
		pass

	"""passed if just running agent on scenarios (only trainer implements this)"""
	def _incorp_agent_reward(self, agent, state, action, state2, reward_value):
		pass