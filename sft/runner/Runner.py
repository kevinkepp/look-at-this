import os
import pkgutil
from importlib import import_module

from abc import abstractmethod

import numpy as np
import threading
import time
import random
import sys
import theano

from sft.Actions import Actions
from sft.State import State
from sft.sim.Simulator import Simulator


class Runner(object):
	WORLD_CONFIG_NAME = "world"
	AGENT_CONFIG_NAME_DIR = "agents"
	PERSIST_MODEL_EVERY = 100

	def run(self, experiment, threaded=False):
		self.set_seed()
		world_config, agent_configs = self.get_configs(experiment)
		scenarios = self.init_scenarios(world_config)
		if threaded:
			threads = []
			for agent in agent_configs:
				thread = threading.Thread(target=self.run_agent, args=(agent, scenarios))
				thread.daemon = False
				thread.start()
				threads.append(thread)
			for t in threads:
				t.join()
		else:
			for agent in agent_configs:
				self.run_agent(agent, scenarios)
		world_config.world_logger.close_files()

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
		scenarios = []
		for n in range(config.epochs):
			scenario = config.world_gen.get_next()
			scenarios.append(scenario)
			config.sampler.next_epoch()
			config.world_logger.log_init_state_and_world(scenario.world, scenario.pos)
			config.world_logger.next_epoch()
			config.sampler.next_epoch()
		return scenarios

	def run_agent(self, config, scenarios):
		logger = config.logger
		logger.log_message("{0} - Start training over {1} epochs".format(config.__name__, config.epochs))
		sim = Simulator(config.view_size)
		time_start = time.time()
		for n in range(len(scenarios)):
			scenario = scenarios[n]
			success, hist = self.run_epoch(config, sim, n, scenario)
			logger.log_results(hist, success)
			logger.log_message("{0} - Epoch {1} - Success: {2} - Steps: {3}".format(config.__name__, n, success, len(hist)))
			if n % self.PERSIST_MODEL_EVERY == 0:
				logger.log_model(config.model, str(n))
			logger.next_epoch()
			config.agent.new_episode()
		time_diff = time.time() - time_start
		logger.log_model(config.model)
		logger.log_message("{0} - Finished training, took {1} seconds".format(config.__name__, time_diff))
		logger.close_files()

	def run_epoch(self, config, sim, epoch, scenario):
		"""Run training episode with given initial scenario"""
		sim.initialize(scenario.world, scenario.pos)
		eps = self._get_eps(config, epoch)
		config.logger.log_parameter("epsilon", eps)
		hist = []
		while len(hist) < config.max_steps:
			# result is 1 for on target, 0 for oob and -1 for non-terminal
			res = self.run_step(hist, sim, config.action_hist_len, config.agent, config.reward, eps)
			if res != -1:
				return res, hist
		# if max steps are exceeded episode was not successful
		return 0, hist

	def run_step(self, hist, sim, state_action_len, agent, reward, eps):
		"""Conducts training step and returns 1 for success, 0 for loss and -1 for non-terminal"""
		view = sim.get_current_view()
		state = self.get_state(view, hist, state_action_len)
		action = agent.choose_action(state, eps)
		hist.append(action)
		view2 = sim.update_view(action)
		reward_value = reward.get_reward(view, view2)
		oob = sim.is_oob()
		at_target = sim.is_at_target()
		# epoch ends when agent runs out of bounds or hits the target
		terminal = oob or at_target
		# if new state is terminal None will be given to agent
		state2 = self.get_state(view2, hist, state_action_len) if not terminal else None
		self._incorp_agent_reward(agent, state, action, state2, reward_value)
		if at_target:
			return 1
		elif oob:
			return 0
		else:
			return -1

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

	def get_state(self, view, action_hist, state_action_hist_len):
		actions = np.zeros([state_action_hist_len, len(Actions.all)], dtype=theano.config.floatX)
		# take last n actions, this will be smaller or empty if there are not enough actions
		last_actions = action_hist[-state_action_hist_len:] if state_action_hist_len > 0 else []
		for i in range(len(last_actions)):
			action = last_actions[i]
			actions[i] = Actions.get_one_hot(action)
		return State(view, actions)

	def set_seed(self, seed=None):
		if seed is None:
			# set randomly generated random seed for comparability of agents run in this execution
			# while maintain the generation of different paths and agent behaviour for different executions
			seed = random.randint(0, sys.maxint)
		random.seed(seed)

	"""later used by Trainer to run epsilon update and by Tester to set epsilon to wished value"""
	@abstractmethod
	def _get_eps(self, config, epoch):
		pass

	"""passed if just running agent on scenarios (only trainer implements this)"""
	def _incorp_agent_reward(self, agent, state, action, state2, reward_value):
		pass