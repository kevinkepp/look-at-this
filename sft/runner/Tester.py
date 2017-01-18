import os, shutil
import pkgutil
from importlib import import_module
from sft.sim.PathWorldLoader import PathWorldLoader

from sft.runner.Runner import Runner


class Trainer(Runner):

	EPSILON = 0
	TESTSET_INIT_POS_FILE = "init_states.tsv"
	# values and symbols for reading the by us created state input file
	PATH_VALUE = 0.588235294118
	TARGET_VALUE = 1
	SYMBOL_PATH = 1
	SYMBOL_TARGET = 2
	SYMBOL_EMPTY = 0
	TEST_WORLD_PATH = "sft/config-test/world.py"
	TEST_AGENT_PATH = "sft/config-test/agents"
	"""
	00100
	00100
	00200
	02220
	00200
	"""

	def _load_model(self):
		"""load one specific agent and model of agent"""
		pass

	def _load_exp_models(self, exp_path):
		"""loads the agents and corresponding models of each agent"""
		pass

	def _load_worlds(self, testset_worlds_path):
		"""load worlds and init states to run agent models on"""
		pwl = PathWorldLoader(None, testset_worlds_path, self.view_size, self.world_size, None, path_init_file=testset_worlds_path + "/" + self.TESTSET_INIT_POS_FILE)
		amt_worlds = len([name for name in os.listdir('.') if os.path.isfile(name)]) - 1
		for i in range(amt_worlds):
			self.scenarios.append(pwl.get_next(random_choice=False))
		shutil.copytree(testset_worlds_path, ...)

	def run_on_exp(self, exp_path, agent_dict, testset_worlds_path):
		pass

	def run_one_model(self, model_path, agent_config_path, world_config_path, testset_worlds_path):
		pass

	def _copy_config_files(self, world_config_path, agent_config_path_arr):
		self._delete_old_config_files()
		shutil.copy(world_config_path, self.TEST_WORLD_PATH)
		for agent in agent_config_path_arr:
			agent_name = agent.split("/")[-1]
			shutil.copy(agent, self.TEST_AGENT_PATH + "/" + agent_name)

	def _delete_old_config_files(self):
		if os.path.isfile(self.TEST_WORLD_PATH):
			os.remove(self.TEST_WORLD_PATH)
		for f in os.listdir(self.TEST_AGENT_PATH):
			if f != "__init__.py":
				os.remove(self.TEST_AGENT_PATH + "/" + f)

	def _get_eps(self, config, epoch):
		return self.EPSILON

	def _create_state_from_file(self, state_file_path):
		pass

	def get_q_one_state(self, state_path):
		pass

	def __init__(self):
		# TODO
		# copy worlds and init to folder tester/worlds
		# load agent
		# init AgentTesterLogger
		# overwrite agentconfig.agent.logger = AgentTesterLogger
		# overwrite config.logger = AgentTesterLogger
		# load model and update agent with it
		# perhaps adjust max steps
		self.scenarios = []

	def run(self, experiment, testset_worlds_path):
		self.set_seed()
		world_config, agent_configs = self.get_configs(experiment)
		scenarios = self.init_scenarios(world_config)
		for agent in agent_configs:
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
