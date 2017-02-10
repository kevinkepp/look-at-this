import os
from shutil import copyfile

import threading

from sft.runner.Runner import Runner


class Trainer(Runner):
	def run(self, experiment, threaded=False):
		world_config = self.get_world_config(experiment)
		scenarios = self.init_scenarios(world_config)
		nb_agent_runs = max(1, world_config.nb_agent_runs)
		self.copy_agent_configs(experiment, nb_agent_runs)
		agent_configs = self.get_agent_configs(experiment)
		# generate distinct seeds for every agent copy and repeat this list for every distinct agent
		seeds = self.gen_seeds(nb_agent_runs) * (len(agent_configs) / nb_agent_runs)
		self.run_agents(world_config, agent_configs, scenarios, seeds, threaded)
		self.reset_agent_configs(experiment)

	def get_agent_dir(self, exp_module):
		exp_path = exp_module.__file__
		# check if running from .pyc file and change path to .py
		if exp_path.endswith("pyc"):
			exp_path = exp_path[:-1]
		return exp_path[:-len("__init__.py")] + "agents"

	def get_agent_files(self, exp_module):
		agent_dir = self.get_agent_dir(exp_module)
		return [agent_dir + "/" + a for a in os.listdir(agent_dir) if
					   a != "__init__.py" and not a.endswith("pyc")]  # [:-3] to remove ".py"

	def copy_agent_configs(self, exp_module, nb_action_runs):
		agent_files = self.get_agent_files(exp_module)
		# create nb_action_runs instances of every agent
		for agent_file in agent_files:
			# copy agent files
			for i in range(1, nb_action_runs):
				copyfile(agent_file, "%s_%d.py" % (agent_file[:-3], i))
			# for the first instance just rename the original file
			os.rename(agent_file, agent_file[:-3] + "_0.py")

	def reset_agent_configs(self, exp_module):
		agent_files = self.get_agent_files(exp_module)
		for agent_file in agent_files:
			if agent_file.endswith("_0.py"):
				os.rename(agent_file, agent_file[:-5] + ".py")
			else:
				os.remove(agent_file)
		# also delete .pyc files
		agent_dir = self.get_agent_dir(exp_module)
		for f in os.listdir(agent_dir):
			if f.endswith("pyc"):
				os.remove(agent_dir + "/" + f)

	def run_agents(self, world_config, agent_configs, scenarios, seeds, threaded):
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
			for agent, seed in zip(agent_configs, seeds):
				self.run_agent(agent, scenarios, seed)
		world_config.world_logger.close_files()

	def init_scenarios(self, world_config):
		seed = self.set_seed()
		world_config.world_logger.log_message("Using seed %s for initializing scenarios" % str(seed))
		scenarios = []
		for n in range(world_config.epochs):
			scenario = world_config.world_gen.get_next()
			scenarios.append(scenario)
			world_config.sampler.next_epoch()
			world_config.world_logger.log_init_state_and_world(scenario.world, scenario.pos)
			world_config.world_logger.next_epoch()
		return scenarios

	def _get_eps(self, config, epoch):
		eps = config.epsilon_update.get_value(epoch)
		return eps

	def _incorp_agent_reward(self, agent, state, action, state2, reward_value):
		agent.incorporate_reward(state, action, state2, reward_value)
