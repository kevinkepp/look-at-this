import os
import cv2
import numpy as np

from sft.log.Logger import BaseLogger


class WorldLogger(BaseLogger):
	""" used to log data (like parameters, configuration, ...) in order to enable proper experimentation """

	def __init__(self, world_cfg_path, config_name):
		super(WorldLogger, self).__init__()
		# default names of the files and folders
		self.name_folder_world = "world"
		self.name_folder_world_init = "world_init_logs"
		self.name_file_init = "init_states" + self.FILE_SUFFIX_LOGS
		self.name_file_cfg_world = "world" + self.FILE_SUFFIX_CFG
		self.name_setup = self._get_exp_name(config_name)

		self.file_init_states = None  # file for log the init states

		# self._get_name_from_config_file(agent_cfg_path)
		self.curr_exp_log_dir = self.log_dir
		self._create_folders()
		self._copy_config_file(world_cfg_path, self.name_file_cfg_world)

	def _get_exp_name(self, agent_name):
		"""extract the name of the experiment for naming a folder later"""
		return agent_name.split(".")[2]

	def get_exp_log_path(self):
		"""returns the path to the experiment log folder"""
		return self.curr_exp_log_dir

	def _create_folders(self):
		""" creates the folder structure for the current experiment """
		if not os.path.exists(self.curr_exp_log_dir):
			os.makedirs(self.curr_exp_log_dir)
		# create folder for current experiment
		folder_name = self._get_timestamp() + "_" + self.name_setup
		dir_path = self.curr_exp_log_dir + "/" + folder_name
		os.makedirs(dir_path)
		self.curr_exp_log_dir = dir_path
		# create world log dir
		dir_path += "/" + self.name_folder_world
		os.makedirs(dir_path)
		self.log_dir = dir_path
		# create folder for saving the parameter files
		os.makedirs(self.log_dir + "/" + self.NAME_FOLDER_PARAMETERS)

	def log_init_state_and_world(self, world_state, agent_pos):
		""" saves initial state and world-configuration """
		if self.file_init_states is None:
			os.makedirs(self.log_dir + "/" + self.name_folder_world_init)
			path = self.log_dir + "/" + self.name_folder_world_init + "/" + self.name_file_init
			self.file_init_states = self.open_file(path)
			self.log_message("created logfile and folder for init states and world states")
			headline = "{}\t{}\t{}\n".format("epoch", "agent-init-x", "agent-init-y")
			self.file_init_states.write(headline)
		# log the init state and view dims in a logfile
		init_state_text_line = "{}\t{}\t{}\n".format(self.epoch, agent_pos.x, agent_pos.y)
		self.file_init_states.write(init_state_text_line)
		# log the worldstate as an grayscale png image
		img = (world_state * 255.9).astype(np.uint8)
		img_file_name = "epoch{}_worldstate.png".format(self.epoch)
		path = self.log_dir + "/" + self.name_folder_world_init + "/" + img_file_name
		cv2.imwrite(path, img)
