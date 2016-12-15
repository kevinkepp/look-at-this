import os
import cv2
import numpy as np
import datetime as dt
from shutil import copyfile


class WorldLogger:
	""" used to log data (like parameters, configuration, ...) in order to enable proper experimentation """

	def __init__(self, world_cfg_path, config_name):
		# default names of the files and folders
		self.file_suffix_logs = ".tsv"
		self.file_suffix_cfg = ".py"
		self.general_log_dir = "tmp/logs"  # later is replaced with dir of current experiment
		self.name_folder_world = "world"
		self.name_folder_cfg = "config-sample"
		self.name_folder_parameters = "parameter_logs"
		self.name_folder_world_init = "world_init_logs"
		self.name_file_init = "init_states" + self.file_suffix_logs
		self.name_file_messages = "messages" + self.file_suffix_logs
		self.name_file_cfg_world = "world" + self.file_suffix_cfg
		self.name_setup = self._get_exp_name(config_name)

		self.epoch = 0
		self.files_params = {}  # dictionary with all parameter files in it
		self.file_messages = None  # file for logging the general messages
		self.file_init_states = None  # file for logging the init states
		self.world_log_dir = None  # path to the folder where the world stuff is logged

		# self._get_name_from_config_file(agent_cfg_path)
		self._create_folders()
		self._copy_config_file(world_cfg_path)

	def open_file(self, path):
		buffering = 1  # 1 means line buffering
		return open(path, 'w', buffering)

	def next_epoch(self):
		""" increases epoch, which is used for logging """
		self.epoch += 1

	def _get_exp_name(self, agent_name):
		"""extract the name of the experiment for naming a folder later"""
		return agent_name.split(".")[2]

	def get_exp_log_path(self):
		"""returns the path to the experiment log folder"""
		return self.general_log_dir

	def _get_timestamp(self):
		""" creates a timestamp that can be used to log """
		now = dt.datetime.now()
		return now.strftime("%Y%m%d-%H%M%S")

	def _copy_config_file(self, world_cfg_path):
		""" makes a copy of the configfiles to the logging folder """
		copyfile(world_cfg_path, self.world_log_dir + "/" + self.name_file_cfg_world)

	def _create_folders(self):
		""" creates the folder structure for the current experiment """
		if not os.path.exists(self.general_log_dir):
			os.makedirs(self.general_log_dir)
		# create folder for current experiment
		folder_name = self._get_timestamp() + "_" + self.name_setup
		dir_path = self.general_log_dir + "/" + folder_name
		os.makedirs(dir_path)
		self.general_log_dir = dir_path
		# create world log dir
		dir_path += "/" + self.name_folder_world
		os.makedirs(dir_path)
		self.world_log_dir = dir_path
		# create folder for saving the parameter files
		os.makedirs(self.world_log_dir + "/" + self.name_folder_parameters)

	def log_parameter(self, para_name, para_val):
		""" logs a parameter value to a file """
		if para_name not in self.files_params:
			path = self.world_log_dir + "/" + self.name_folder_parameters + "/" + para_name + self.file_suffix_logs
			self.files_params[para_name] = self.open_file(path)
			self.log_message("created parameter logfile for '{}'".format(para_name))
			self.files_params[para_name].write("epoch\tparameter-value\n")
		self.files_params[para_name].write("{}\t{}\n".format(self.epoch, para_val))

	def log_message(self, message):
		""" logs a message (e.g. cloned network) to a general logfile """
		if self.file_messages is None:
			path = self.world_log_dir + "/" + self.name_file_messages
			self.file_messages = self.open_file(path)
			self.file_messages.write(self._create_line_for_msg_logfile("created this logfile"))
		self.file_messages.write(self._create_line_for_msg_logfile(message))

	def _create_line_for_msg_logfile(self, message):
		""" adds timestamp, tab and succeeding newline operator"""
		return self._get_timestamp() + "\t" + message + "\n"

	def log_init_state_and_world(self, world_state, agent_pos):
		""" saves initial state and world-configuration """
		if self.file_init_states is None:
			os.makedirs(self.world_log_dir + "/" + self.name_folder_world_init)
			path = self.world_log_dir + "/" + self.name_folder_world_init + "/" + self.name_file_init
			self.file_init_states = self.open_file(path)
			self.log_message("created logfile and folder for init states and world states")
			headline = "{}\t{}\t{}\n".format("epoch", "agent-init-x", "agent-init-y")
			self.file_init_states.write(headline)
		# logging the init state and view dims in a logfile
		init_state_text_line = "{}\t{}\t{}\n".format(self.epoch, agent_pos.x, agent_pos.y)
		self.file_init_states.write(init_state_text_line)
		# logging the worldstate as an grayscale png image
		img = (world_state * 255.9).astype(np.uint8)
		img_file_name = "epoch{}_worldstate.png".format(self.epoch)
		path = self.world_log_dir + "/" + self.name_folder_world_init + "/" + img_file_name
		cv2.imwrite(path, img)

	# TODO: include closing of files method to clean up!
	def end_logging(self):
		""" ends logging and closes all open files """
		for each_file in self.files_params.values():
			each_file.close()
		self.log_message("ending logging and closed files, now also closing this log file")
		self.file_init_states.close()
		self.file_messages.close()
