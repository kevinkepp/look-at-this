import os

import datetime as dt
from shutil import copyfile


class Logger:
	""" used to log data (like parameters, configuration, ...) in order to enable proper experimentation """
	def __init__(self, configfile_path):
		# default names of the files and folders
		self.file_suffix_logs = ".log"
		self.file_suffix_cfg = ".cfg"
		self.general_log_dir = "tmp/logs"  # later is replaced with dir of current experiment
		self.name_folder_parameters = "parameter_logs"
		self.name_file_messages = "messages" + self.file_suffix_logs
		self.name_file_cfg = "configuration" + self.file_suffix_cfg
		self.name_file_results = "results" + self.file_suffix_logs

		self.epoch = 1
		self.files_params = {}  # dictionary with all parameter files in it
		self.file_messages = None  # file for logging the general messages
		self.file_results = None  # file for logging the results
		self.name_setup = ""  # later stores the name of the folder to be created

		self._get_name_from_config_file(configfile_path)
		self._create_folders()
		self._copy_config_file(configfile_path)

	def next_epoch(self):
		""" increases epoch, which is used for logging """
		self.epoch += 1  # TODO: talk about convention if we start with 0 or 1 and when to increase

	def _get_timestamp(self):
		""" creates a timestamp that can be used to log """
		now = dt.datetime.now()
		return now.strftime("%Y%m%d-%H%M%S")

	def _get_name_from_config_file(self, configfile_path):
		"""  use config file for naming folder """
		# TODO: get proper name
		self.name_setup = "something from the config file"

	def _copy_config_file(self, configfile_path):
		""" makes a copy of the configfile to the logging folder """
		copyfile(configfile_path, self.general_log_dir + "/" + self.name_file_cfg)

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

	def log_parameter(self, para_name, para_val):
		""" logs a parameter value to a file """
		if para_name not in self.files_params:
			path = self.general_log_dir + "/" + self.name_folder_parameters + "/" + para_name + self.file_suffix_logs
			self.files_params[para_name] = open(path , 'w')
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

	def _create_line_for_msg_logfile(self,message):
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

	def log_init_state_and_world(self, world_state, agent_pos_i, agent_pos_j, view_dims):
		""" saves initial state and world-configuration """
		# TODO: code logging of init state and world state for later use
		pass

	def end_logging(self):  # TODO: include this to clean up!
		""" ends logging and closes all open files """
		for each_file in self.files_params.values():
			each_file.close()
		self.file_results.close()
		self.log_message("ending logging and closed files, now also closing this log file")
		self.file_messages.close()
