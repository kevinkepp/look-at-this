import os

import sys

from sft.log.Logger import BaseLogger


class AgentLogger(BaseLogger):
	""" used to log data (like parameters, configuration, ...) in order to enable proper experimentation """
	LOG_AGENT_PREFIX = "agent"
	NAME_MODELS_FOLDER = "models"
	NAME_MODEL_PREFIX = "model"
	FILE_SUFFIX_MODEL = ".h5"
	FILE_NAME_RESULTS = "results"

	def __init__(self, agent_module_name):
		agent_module = sys.modules[agent_module_name]
		agent_cfg_path = agent_module.__file__
		# check if running from .pyc file and change path to .py
		if agent_cfg_path.endswith("pyc"):
			agent_cfg_path = agent_cfg_path[:-1]
		super(AgentLogger, self).__init__()
		if agent_module.world.world_logger is not None:
			exp_log_folder = agent_module.world.world_logger.get_exp_log_path()
			self.log_dir = exp_log_folder  # later is replaced with dir of current experiment
		# default names of the files and folders
		self.file_suffix_model = self.FILE_SUFFIX_MODEL
		self.name_folder_models = self.NAME_MODELS_FOLDER
		self.name_file_cfg_agent = self.LOG_AGENT_PREFIX + self.FILE_SUFFIX_CFG
		self.name_file_results = "results" + self.FILE_SUFFIX_LOGS
		self.name_file_actions_taken = "actions" + self.FILE_SUFFIX_LOGS
		self.name_file_model = self.NAME_MODEL_PREFIX
		self.name_setup = agent_module_name.split(".")[-1]

		self.file_results = None  # file for log the results
		self.file_actions_taken = None  # file for log the actions taken

		# self._get_name_from_config_file(agent_cfg_path)
		self._create_folders()
		self._copy_config_file(agent_cfg_path, self.name_file_cfg_agent)

	def _create_folders(self):
		""" creates the folder structure for the current experiment """
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		# create folder for current experiment
		folder_name = self.name_setup
		dir_path = self.log_dir + "/" + self.LOG_AGENT_PREFIX + "_" + folder_name
		os.makedirs(dir_path)
		self.log_dir = dir_path
		# create folder for saving the parameter files
		param_path = self.log_dir + "/" + self.NAME_FOLDER_PARAMETERS
		if not os.path.exists(param_path):
			os.makedirs(self.log_dir + "/" + self.NAME_FOLDER_PARAMETERS)

	def log_results(self, actions_taken, success):
		""" log the results (actions taken and success-bool) and close the files of this experiment"""
		if self.file_results is None:
			# create result file
			path = self.log_dir + "/" + self.name_file_results
			self.file_results = self.open_file(path)
			self.log_message("created results logfile")
			self.file_results.write("epoch\tsuccess\t#actions-taken\n")
			# create actions-taken file
			path = self.log_dir + "/" + self.name_file_actions_taken
			self.file_actions_taken = self.open_file(path)
			self.log_message("created actions-taken logfile")
			self.file_actions_taken.write("epoch\tactions-taken\n")
		self.file_actions_taken.write("{}\t{}\n".format(self.epoch, actions_taken))
		self.file_results.write("{}\t{}\t{}\n".format(self.epoch, success, len(actions_taken)))

	def log_model(self, model, name=None):
		""" log model for later analysis """
		# create directory for saving the models
		path = self.log_dir + "/" + self.name_folder_models
		if not os.path.exists(path):
			os.makedirs(path)
		path = path + "/" + self.name_file_model
		if name is not None:
			path += "_" + name
		path += self.file_suffix_model
		model.save(path)
