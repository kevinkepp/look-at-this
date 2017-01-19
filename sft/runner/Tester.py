import os, shutil
import pkgutil
from importlib import import_module
import numpy as np
import theano
from sft.State import State

from sft.Actions import Actions
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
	AGENT_DUMMY_LOGGER_NAME ="DummyLogger"
	WORLD_LOGGER_INIT = "WorldLogger(__name__)"
	TEST_MODULE_NAME = "config_test"
	TESTER_OUTPUT_FOLDER_NAME = "tester"

	# TODO
	def run_model(self, model_path, agent_config_path, world_config_path, testset_worlds_path):
		pass

	def _load_worlds(self, testset_worlds_path, world_size, view_size):
		"""load worlds and init states to run agent models on"""
		scenarios = []
		pwl = PathWorldLoader(None, testset_worlds_path, view_size, world_size, None, path_init_file=testset_worlds_path + "/" + self.TESTSET_INIT_POS_FILE)
		amt_worlds = len(os.listdir(testset_worlds_path)) - 1
		for i in range(amt_worlds):
			scenarios.append(pwl.get_next(random_choice=False))
		return scenarios

	def _copy_world_config(self, world_config_folder_path):
		if os.path.isfile(self.TEST_WORLD_PATH):
			os.remove(self.TEST_WORLD_PATH)
		shutil.copy(world_config_folder_path, self.TEST_WORLD_PATH)

	def run_exp(self, exp_path, testset_worlds_path):
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
						ep = int(fm.split(AgentLogger.FILE_SUFFIX_MODEL)[0][len(AgentLogger.NAME_MODEL_PREFIX) + 1:])  # + 1 for the underscore
						model_src_paths[ep] = mdl_path + "/" + fm
				agent_models_src_paths.append(model_src_paths)
		world_config_folder_path = exp_path + "/" + WorldLogger.NAME_FOLDER_WORLD + "/" + WorldLogger.NAME_FOLDER_WORLD + BaseLogger.FILE_SUFFIX_CFG
		self._copy_world_config(world_config_folder_path)
		scenarios = None
		# run the stuff
		for i_a in range(len(agent_config_paths)):
			for ep in agent_models_src_paths[i_a].keys():
				# replace the input to AgentTesterLogger in agent config files
				agent_tmp_path = self._copy_agent_config_file(agent_config_paths[i_a], ep)
				replace_in_file(agent_tmp_path, self.AGENT_TESTER_LOGGER_NAME + "(__name__)",
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

	def _copy_agent_config_file(self, agent_config_path, ep, agent_logger_replacement_name):
		self._delete_old_agent_config_files()
		agent_name = agent_config_path.split("/")[-2][len(AgentLogger.LOG_AGENT_PREFIX) + 1:] + "_" + str(ep) + ".py"
		shutil.copy(agent_config_path, self.TEST_AGENT_PATH + "/" + agent_name)
		self._replace_loggers("/".join(self.TEST_WORLD_PATH.split("/")[:-1]), agent_logger_replacement_name)
		file_path = self.TEST_AGENT_PATH + "/" + agent_name
		return file_path

	def _delete_old_agent_config_files(self):
		for f in os.listdir(self.TEST_AGENT_PATH):
			if f != "__init__.py":
				os.remove(self.TEST_AGENT_PATH + "/" + f)

	def _get_eps(self, config, epoch):
		return self.EPSILON

	def _replace_loggers(self, exp_path, agent_logger_replacement_name):
		self._replace_world_logger(exp_path + "/world.py")
		for agent_file in os.listdir(exp_path + "/agents"):
			self._replace_agent_logger(exp_path + "/agents/" + agent_file, agent_logger_replacement_name)

	def _replace_world_logger(self, file_path):
		replace_in_file(file_path, self.WORLD_LOGGER_INIT, "None")

	def _replace_agent_logger(self, file_path, logger_replacement_name):
		replace_in_file(file_path, self.AGENT_LOGGER_NAME, logger_replacement_name)

	"""
	def _replace_agent_logger(self, file_path, tester_path, model_epoch):
		search = "AgentLogger(__name__)"
		replace = "AgentTesterLogger(__name__, " + tester_path + ", " + model_epoch + ")"
		replace_in_file(file_path, search, replace)
	"""

	# TODO: load state from file, predict Q, print Q, create plot
	# values and symbols for reading the by us created state input file
	state_file_transfer_dict = {
		0: 0,
		1: 0.588235294118,  # path value
		2: 1  # target value
	}
	"""
	00100
	00100
	00200
	02220
	00200
	down <- newest
	down
	up
	left
	"""

	def _create_state_from_file(self, state_file_path):
		state_file = open(state_file_path, 'r')
		w = 0
		h = 0
		view_lines = []
		ah = []
		for line in state_file:
			if any(char.isdigit() for char in line):
				if w == 0:
					w = len(line) - 1
				view_lines.append(line.rstrip("\n"))
			else:
				ah.append(Actions.get_by_name(line.strip("\n")))
		# create view
		v = []
		for line in view_lines:
			l = []
			for s in line:
				l.append(self.state_file_transfer_dict[int(s)])
			v.append(l)
		state_view = np.array(v, dtype=theano.config.floatX)
		# create state
		return self.get_state(state_view, ah, len(ah))

	def get_q_one_state(self, state_path, world_config_path, agent_path, model_path):
		in_state = self._create_state_from_file(state_path)
		self._copy_world_config(world_config_path)
		agent_tmp_path = self._copy_agent_config_file(agent_path, 0, self.AGENT_DUMMY_LOGGER_NAME)
		replace_in_file(agent_tmp_path, self.AGENT_DUMMY_LOGGER_NAME + "(__name__)",
						self.AGENT_DUMMY_LOGGER_NAME + "()")
		exp = import_module("." + self.TEST_MODULE_NAME, "sft")
		_, agent_config = self.get_configs(exp)
		# load model
		agent_config[0].agent.model.load(model_path)
		# predict q values
		qs = agent_config[0].agent.model.predict_qs(in_state.view, in_state.actions)
		print(Actions.names)
		print(qs)
		return qs
