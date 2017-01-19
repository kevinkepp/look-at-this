import os
from shutil import copyfile

from sft.log.AgentLogger import AgentLogger


class AgentTesterLogger(AgentLogger):
	""" used to log data (like parameters, configuration, ...) in order to enable proper experimentation """

	def __init__(self, agent_module_name, tester_path, model_epoch):
		self.tester_path = tester_path
		self.model_epoch = model_epoch
		super(AgentTesterLogger, self).__init__(agent_module_name)

	"""copy agent config file once"""
	def _copy_config_file(self, cfg_file_path, file_name):
		""" makes a copy of the configfiles to the log folder"""
		file_path = self.agent_dir_path + "/" + file_name
		if not os.path.isfile(file_path):
			copyfile(cfg_file_path, file_path)

	def _create_folders(self):
		""" creates the folder structure for the current tested model """
		if not os.path.exists(self.tester_path):
			os.makedirs(self.tester_path)
		# create folder for current agent
		agent_folder_name = "".join((self.name_setup).split("_")[:-1])
		agent_dir_path = self.tester_path + "/" + agent_folder_name
		if not os.path.exists(agent_dir_path):
			os.makedirs(agent_dir_path)
		self.agent_dir_path = agent_dir_path
		# create folder for current trained model of agent
		model_folder_name = str(self.model_epoch).zfill(5)
		model_dir_path = agent_dir_path + "/" + model_folder_name
		if not os.path.exists(model_dir_path):
			os.makedirs(model_dir_path)
		# set log dir to this location, so every stuff gets logged there
		self.log_dir = model_dir_path
		# create folder for saving the parameter files
		param_path = self.log_dir + "/" + self.NAME_FOLDER_PARAMETERS
		if not os.path.exists(param_path):
			os.makedirs(param_path)

	def log_model(self, model):
		"""stores model if it was not already copied to path"""
		file_path = self.log_dir + "/" + self.name_file_model + str(self.model_epoch) + self.file_suffix_model
		if not os.path.isfile(file_path):
			model.save(file_path)
