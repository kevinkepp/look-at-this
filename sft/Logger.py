import os
import cv2
import numpy as np
import datetime as dt
from shutil import copyfile


class Logger:
	""" used to log data (like parameters, configuration, ...) in order to enable proper experimentation """

	def __init__(self, world_cfg_path, agent_cfg_path, agent_name):
		# default names of the files and folders
		self.file_suffix_logs = ".tsv"
		self.file_suffix_cfg = ".py"
		self.general_log_dir = "tmp/logs"  # later is replaced with dir of current experiment
		self.name_folder_cfg = "config"
		self.name_folder_parameters = "parameter_logs"
		self.name_folder_world_init = "world_init_logs"
		self.name_folder_models = "models"
		self.name_file_init = "init_states" + self.file_suffix_logs
		self.name_file_messages = "messages" + self.file_suffix_logs
		self.name_file_cfg_agent = "agent-config" + self.file_suffix_cfg
		self.name_file_cfg_world = "world" + self.file_suffix_cfg
		self.name_file_results = "results" + self.file_suffix_logs
		self.name_setup = agent_name

		self.epoch = 0
		self.files_params = {}  # dictionary with all parameter files in it
		self.file_messages = None  # file for logging the general messages
		self.file_results = None  # file for logging the results
		self.file_init_states = None  # file for logging the init states

		# self._get_name_from_config_file(agent_cfg_path)
		self._create_folders()
		self._copy_config_file(world_cfg_path, agent_cfg_path)

	def next_epoch(self):
		""" increases epoch, which is used for logging """
		self.epoch += 1

	def _get_timestamp(self):
		""" creates a timestamp that can be used to log """
		now = dt.datetime.now()
		return now.strftime("%Y%m%d-%H%M%S")

	def _get_name_from_config_file(self, agent_cfg_path):
		"""  use config file for naming folder """
		# TODO: get proper name from config file
		self.name_setup = "something from the config file"

	def _copy_config_file(self, world_cfg_path, agent_cfg_path):
		""" makes a copy of the configfiles to the logging folder """
		copyfile(world_cfg_path, self.general_log_dir + "/" + self.name_folder_cfg + "/" + self.name_file_cfg_world)
		copyfile(agent_cfg_path, self.general_log_dir + "/" + self.name_folder_cfg + "/" + self.name_file_cfg_agent)

	def _create_folders(self):
		""" creates the folder structure for the current experiment """
		if not os.path.exists(self.general_log_dir):
			os.makedirs(self.general_log_dir)
		# create folder for current experiment
		folder_name = self._get_timestamp() + "_" + self.name_setup
		dir_path = self.general_log_dir + "/" + folder_name
		os.makedirs(dir_path)
		self.general_log_dir = dir_path
		# create folder for saving the parameter files
		os.makedirs(self.general_log_dir + "/" + self.name_folder_parameters)
		# config dir
		os.makedirs(self.general_log_dir + "/" + self.name_folder_cfg)

	def log_parameter(self, para_name, para_val):
		""" logs a parameter value to a file """
		if para_name not in self.files_params:
			path = self.general_log_dir + "/" + self.name_folder_parameters + "/" + para_name + self.file_suffix_logs
			self.files_params[para_name] = open(path, 'w')
			self.log_message("created parameter logfile for '{}'".format(para_name))
			self.files_params[para_name].write("epoch\tparameter-value\n")
		self.files_params[para_name].write("{}\t{}\n".format(self.epoch, para_val))

	def log_message(self, message):
		""" logs a message (e.g. cloned network) to a general logfile """
		if self.file_messages is None:
			path = self.general_log_dir + "/" + self.name_file_messages
			self.file_messages = open(path, 'w')
			self.file_messages.write(self._create_line_for_msg_logfile("created this logfile"))
		self.file_messages.write(self._create_line_for_msg_logfile(message))

	def _create_line_for_msg_logfile(self, message):
		""" adds timestamp, tab and succeeding newline operator"""
		return self._get_timestamp() + "\t" + message + "\n"

	def log_results(self, actions_taken, success):
		""" log the results (actions taken and success-bool) and close the files of this experiment"""
		if self.file_results is None:
			path = self.general_log_dir + "/" + self.name_file_results
			self.file_results = open(path, 'w')
			self.log_message("created results logfile")
			self.file_results.write("epoch\tsuccess\tactions-taken\n")
		self.file_results.write("{}\t{}\t{}\n".format(self.epoch, success, actions_taken))

	def log_init_state_and_world(self, world_state, agent_pos):
		""" saves initial state and world-configuration """
		if self.file_init_states is None:
			os.makedirs(self.general_log_dir + "/" + self.name_folder_world_init)
			path = self.general_log_dir + "/" + self.name_folder_world_init + "/" + self.name_folder_world_init
			self.file_init_states = open(path, 'w')
			self.log_message("created logfile and folder for init states and world states")
			headline = "{}\t{}\t{}\n".format("epoch", "agent-init-x", "agent-init-y")
			self.file_init_states.write(headline)
		# logging the init state and view dims in a logfile
		init_state_text_line = "{}\t{}\t{}\n".format(self.epoch, agent_pos.x, agent_pos.y)
		self.file_init_states.write(init_state_text_line)
		# logging the worldstate as an grayscale png image
		img = (world_state * 255.9).astype(np.uint8)
		img_file_name = "epoch{}_worldstate.png".format(self.epoch)
		path = self.general_log_dir + "/" + self.name_folder_world_init + "/" + img_file_name
		# img = Image.fromarray(img)
		# img.save(path)
		cv2.imwrite(path, img)

	def log_model(self, model):
		""" log model for later analysis """
		# TODO: save model somehow
		# create directory for saving the models
		path = self.general_log_dir + "/" + self.name_folder_models
		if not os.path.exists(path):
			os.makedirs(path)

	# TODO: include closing of files method to clean up!
	def end_logging(self):
		""" ends logging and closes all open files """
		for each_file in self.files_params.values():
			each_file.close()
		self.file_results.close()
		self.log_message("ending logging and closed files, now also closing this log file")
		self.file_messages.close()
