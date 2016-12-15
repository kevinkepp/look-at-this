import os
import cv2
import numpy as np
import datetime as dt
from shutil import copyfile


class Logger:
	""" used to log data (like parameters, configuration, ...) in order to enable proper experimentation """

	def __init__(self, agent_cfg_path, agent_name, exp_log_folder):
		# default names of the files and folders
		self.file_suffix_logs = ".tsv"
		self.file_suffix_cfg = ".py"
		self.file_suffix_model = ".h5"
		self.general_log_dir = exp_log_folder  # later is replaced with dir of current experiment
		self.name_folder_cfg = "config-sample"
		self.name_folder_parameters = "parameter_logs"
		self.name_folder_world_init = "world_init_logs"
		self.name_folder_models = "models"
		self.name_file_init = "init_states" + self.file_suffix_logs
		self.name_file_messages = "messages" + self.file_suffix_logs
		self.name_file_cfg_agent = "agent" + self.file_suffix_cfg
		self.name_file_cfg_world = "world" + self.file_suffix_cfg
		self.name_file_results = "results" + self.file_suffix_logs
		self.name_file_actions_taken = "actions" + self.file_suffix_logs
		self.name_file_model = "model"
		self.name_setup = agent_name.split(".")[-1]

		self.epoch = 0
		self.files_params = {}  # dictionary with all parameter files in it
		self.file_messages = None  # file for logging the general messages
		self.file_results = None  # file for logging the results
		self.file_actions_taken = None  # file for logging the actions taken
		self.file_init_states = None  # file for logging the init states

		# self._get_name_from_config_file(agent_cfg_path)
		self._create_folders()
		self._copy_config_file(agent_cfg_path)

	# def get_dir_path(self, *dirs):
	# 	assert len(dirs) > 0
	# 	# append all directory names
	# 	path = os.path.join(dirs[0], *dirs[1:]) if len(dirs) > 1 else dirs[0]
	# 	# make sure directory exists
	# 	os.makedirs(path)
	# 	return path

	def open_file(self, path):
		buffering = 1  # 1 means line buffering
		return open(path, 'w', buffering)

	def next_epoch(self):
		""" increases epoch, which is used for logging """
		self.epoch += 1

	def _get_timestamp(self):
		""" creates a timestamp that can be used to log """
		now = dt.datetime.now()
		return now.strftime("%Y%m%d-%H%M%S")

	def _copy_config_file(self, agent_cfg_path):
		""" makes a copy of the configfiles to the logging folder """
		copyfile(agent_cfg_path, self.general_log_dir + "/" + self.name_file_cfg_agent)

	def _create_folders(self):
		""" creates the folder structure for the current experiment """
		if not os.path.exists(self.general_log_dir):
			os.makedirs(self.general_log_dir)
		# create folder for current experiment
		folder_name = self.name_setup
		dir_path = self.general_log_dir + "/" + folder_name
		os.makedirs(dir_path)
		self.general_log_dir = dir_path
		# create folder for saving the parameter files
		os.makedirs(self.general_log_dir + "/" + self.name_folder_parameters)
		# config-sample dir
		os.makedirs(self.general_log_dir + "/" + self.name_folder_cfg)

	def log_parameter(self, para_name, para_val):
		""" logs a parameter value to a file """
		if para_name not in self.files_params:
			path = self.general_log_dir + "/" + self.name_folder_parameters + "/" + para_name + self.file_suffix_logs
			self.files_params[para_name] = self.open_file(path)
			self.log_message("created parameter logfile for '{}'".format(para_name))
			self.files_params[para_name].write("epoch\tparameter-value\n")
		self.files_params[para_name].write("{}\t{}\n".format(self.epoch, para_val))

	def log_message(self, message):
		""" logs a message (e.g. cloned network) to a general logfile """
		if self.file_messages is None:
			path = self.general_log_dir + "/" + self.name_file_messages
			self.file_messages = self.open_file(path)
			self.file_messages.write(self._create_line_for_msg_logfile("created this logfile"))
		self.file_messages.write(self._create_line_for_msg_logfile(message))

	def _create_line_for_msg_logfile(self, message):
		""" adds timestamp, tab and succeeding newline operator"""
		return self._get_timestamp() + "\t" + message + "\n"

	def log_results(self, actions_taken, success):
		""" log the results (actions taken and success-bool) and close the files of this experiment"""
		if self.file_results is None:
			# create result file
			path = self.general_log_dir + "/" + self.name_file_results
			self.file_results = self.open_file(path)
			self.log_message("created results logfile")
			self.file_results.write("epoch\tsuccess\t#actions-taken\n")
			# create actions-taken file
			path = self.general_log_dir + "/" + self.name_file_actions_taken
			self.file_actions_taken = self.open_file(path)
			self.log_message("created actions-taken logfile")
			self.file_actions_taken.write("epoch\tactions-taken\n")
		self.file_actions_taken.write("{}\t{}\n".format(self.epoch, actions_taken))
		self.file_results.write("{}\t{}\t{}\n".format(self.epoch, success, len(actions_taken)))

	def log_model(self, model):
		""" log model for later analysis """
		# create directory for saving the models
		path = self.general_log_dir + "/" + self.name_folder_models
		if not os.path.exists(path):
			os.makedirs(path)
		path = path + "/" + self.name_file_model + self.file_suffix_model
		model.save(path)

	# TODO: include closing of files method to clean up!
	def end_logging(self):
		""" ends logging and closes all open files """
		for each_file in self.files_params.values():
			each_file.close()
		self.file_results.close()
		self.log_message("ending logging and closed files, now also closing this log file")
		self.file_messages.close()
