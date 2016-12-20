import os
from datetime import datetime
from shutil import copyfile


class BaseLogger(object):
	TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
	FILE_SUFFIX_LOGS = ".tsv"
	FILE_SUFFIX_CFG = ".py"
	OVERALL_LOG_FOLDER = "tmp/logs"
	NAME_FOLDER_PARAMETERS = "parameter_logs"
	NAME_FILE_MESSAGES = "messages" + FILE_SUFFIX_LOGS

	def __init__(self):
		# dictionary storing currently opened files based
		self.open_files = {}
		self.log_dir = self.OVERALL_LOG_FOLDER
		self.epoch = 0
		self.files_params = {}  # dictionary with all parameter files in it
		self.file_messages = None  # file for logging the general messages

	def __del__(self):
		# make sure all files are closed when object is destructed
		if len(self.open_files) > 0:
			self.close_files()

	def open_file(self, path):
		_file = open(path, 'w')
		self.open_files[path] = _file
		return _file

	def next_epoch(self):
		""" increases epoch, which is used for logging """
		self.epoch += 1
		# flush files after each epoch
		self.flush_files()

	def _get_timestamp(self):
		""" creates a timestamp that can be used to log """
		return datetime.now().strftime(self.TIMESTAMP_FORMAT)

	def _copy_config_file(self, cfg_file_path, file_name):
		""" makes a copy of the configfiles to the logging folder """
		copyfile(cfg_file_path, self.log_dir + "/" + file_name)

	def log_parameter(self, para_name, para_val):
		""" logs a parameter value to a file """
		if para_name not in self.files_params:
			path = self.log_dir + "/" + self.NAME_FOLDER_PARAMETERS + "/" + para_name + self.FILE_SUFFIX_LOGS
			self.files_params[para_name] = self.open_file(path)
			self.log_message("created parameter logfile for '{}'".format(para_name))
			self.files_params[para_name].write("epoch\tparameter-value\n")
		self.files_params[para_name].write("{}\t{}\n".format(self.epoch, para_val))

	def log_message(self, message):
		""" logs a message (e.g. cloned network) to a general logfile """
		if self.file_messages is None:
			path = self.log_dir + "/" + self.NAME_FILE_MESSAGES
			self.file_messages = self.open_file(path)
			self.file_messages.write(self._create_line_for_msg_logfile("created this logfile"))
		self.file_messages.write(self._create_line_for_msg_logfile(message))

	def _create_line_for_msg_logfile(self, message):
		""" adds timestamp, tab and succeeding newline operator"""
		return self._get_timestamp() + "\t" + message + "\n"

	def flush_files(self):
		for _, _file in self.open_files.items():
			_file.flush()

	def close_files(self):
		for _, _file in self.open_files.items():
			_file.close()
		self.open_files.clear()
