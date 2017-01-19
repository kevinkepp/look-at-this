import os, shutil
import pkgutil
from importlib import import_module

from sft import replace_in_file
from sft.sim.PathWorldLoader import PathWorldLoader

from sft.runner.Runner import Runner
# for getting the correct folder-names of logged files
from sft.log.AgentLogger import AgentLogger
from sft.log.AgentTesterLogger import AgentTesterLogger
from sft.log.WorldLogger import WorldLogger
from sft.log.Logger import BaseLogger


class Tester(Runner):

	EPSILON = 0
	TESTSET_INIT_POS_FILE = "init_states.tsv"
	TEST_WORLD_PATH = "sft/config_test/world.py"
	TEST_AGENT_PATH = "sft/config_test/agents"
	AGENT_LOGGER_NAME = "AgentLogger"
	AGENT_TESTER_LOGGER_NAME = "AgentTesterLogger"
	WORLD_LOGGER_INIT = "WorldLogger(__name__)"
	TEST_MODULE_NAME = "config_test"
	TESTER_OUTPUT_FOLDER_NAME = "tester"

	# TODO
	def run_one_model(self, model_path, agent_config_path, world_config_path, testset_worlds_path):
		pass

	def _load_worlds(self, testset_worlds_path, world_size, view_size):
		"""load worlds and init states to run agent models on"""
		scenarios = []
		pwl = PathWorldLoader(None, testset_worlds_path, view_size, world_size, None, path_init_file=testset_worlds_path + "/" + self.TESTSET_INIT_POS_FILE)
		amt_worlds = len(os.listdir(testset_worlds_path)) - 1
		for i in range(amt_worlds):
			scenarios.append(pwl.get_next(random_choice=False))
		return scenarios

	def run_on_exp(self, exp_path, testset_worlds_path):
		self.set_seed()
		# getting agent-config paths, saved models of agents path and world-config-path
		agent_config_paths = []
		agent_models_src_paths = []
		for fa in os.listdir(exp_path):
			if AgentLogger.LOG_AGENT_PREFIX in fa:
				ag_path = exp_path + "/" + fa + "/" + AgentLogger.LOG_AGENT_PREFIX + BaseLogger.FILE_SUFFIX_CFG
				agent_config_paths.append(ag_path)
				# model/model_00001.h5 -> extract the epoch
				mdl_path = exp_path + "/" + fa + "/" + AgentLogger.NAME_MODELS_FOLDER
				model_src_paths = {}
				for fm in os.listdir(mdl_path):
					if AgentLogger.NAME_MODEL_PREFIX in fm:
						ep = int(fm.split(AgentLogger.FILE_SUFFIX_MODEL)[0].lstrip(AgentLogger.NAME_MODEL_PREFIX))
						model_src_paths[ep] = mdl_path + "/" + fm
				agent_models_src_paths.append(model_src_paths)
		world_config_folder_path = exp_path + "/" + WorldLogger.NAME_FOLDER_WORLD + "/" + WorldLogger.NAME_FOLDER_WORLD + BaseLogger.FILE_SUFFIX_CFG
		if os.path.isfile(self.TEST_WORLD_PATH):
			os.remove(self.TEST_WORLD_PATH)
		shutil.copy(world_config_folder_path, self.TEST_WORLD_PATH)
		scenarios = None
		# run the stuff
		for i_a in range(len(agent_config_paths)):
			for ep in agent_models_src_paths[i_a].keys():
				# replace the input to AgentTesterLogger in agent config files
				agent_name = self._copy_config_file(agent_config_paths[i_a], ep)
				file_path = self.TEST_AGENT_PATH + "/" + agent_name
				replace_in_file(file_path, self.AGENT_TESTER_LOGGER_NAME + "(__name__)",
								self.AGENT_TESTER_LOGGER_NAME + "(__name__, '" + exp_path + "/" + self.TESTER_OUTPUT_FOLDER_NAME + "', " + str(ep) + ")")
				exp = import_module("." + self.TEST_MODULE_NAME, "sft")
				world_config, agent_config = self.get_configs(exp)
				if scenarios is None:
					scenarios = self._load_worlds(testset_worlds_path, world_config.world_size, world_config.view_size)
				# load model
				agent_config[0].agent.model.load(agent_models_src_paths[i_a][ep])
				# run
				self.run_agent(agent_config[0], scenarios)
		shutil.copytree(testset_worlds_path, exp_path + "/" + self.TESTER_OUTPUT_FOLDER_NAME + "/worlds")

	def _copy_config_file(self, agent_config_path, ep):
		self._delete_old_agent_config_files()
		agent_name = agent_config_path.split("/")[-2].lstrip(AgentLogger.LOG_AGENT_PREFIX + "_") + "_"+str(ep)+".py"
		shutil.copy(agent_config_path, self.TEST_AGENT_PATH + "/" + agent_name)
		self._replace_loggers("/".join(self.TEST_WORLD_PATH.split("/")[:-1]))
		return agent_name

	def _delete_old_agent_config_files(self):
		for f in os.listdir(self.TEST_AGENT_PATH):
			if f != "__init__.py":
				os.remove(self.TEST_AGENT_PATH + "/" + f)

	def _get_eps(self, config, epoch):
		return self.EPSILON

	def _replace_loggers(self, exp_path):
		self._replace_world_logger(exp_path + "/world.py")
		for agent_file in os.listdir(exp_path + "/agents"):
			self._replace_agent_logger(exp_path + "/agents/" + agent_file)

	def _replace_world_logger(self, file_path):
		replace_in_file(file_path, self.WORLD_LOGGER_INIT, "None")

	def _replace_agent_logger(self, file_path):
		replace_in_file(file_path, self.AGENT_LOGGER_NAME, self.AGENT_TESTER_LOGGER_NAME)


	# TODO: load state from file, predict Q, print Q, create plot
	# values and symbols for reading the by us created state input file
	PATH_VALUE = 0.588235294118
	TARGET_VALUE = 1
	SYMBOL_PATH = 1
	SYMBOL_TARGET = 2
	SYMBOL_EMPTY = 0
	""" vorgeschlagenes Format einer Textdatei fuer 5x5 view input und 4 action history. Eine solche Datei muesste nur
	eingelesen werden, uebersetzt werden in ein View [mit den Werten oben] und eine Actionshistory (dazu ist die Action
	Klasse auch schon erweitert .names usw) und dann nur noch aehnlich wie oben laden des Agenten+Modells und draufschieben des Inputs,
	Q Werte erstmal einfach ausgeben oder halt gleich als Bild mit dem Input speichern.
	00100
	00100
	00200
	02220
	00200
	down, down, up, left
	"""


	def _create_state_from_file(self, state_file_path):
		pass

	def get_q_one_state(self, state_path):
		pass
