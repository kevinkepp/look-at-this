import os
from datetime import datetime


class BaseLogger(object):
	TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

	def __init__(self):
		# dictionary storing currently opened files based
		self.open_files = {}

	def __del__(self):
		# make sure all files are closed when object is destructed
		if len(self.open_files) > 0:
			self.close_files()

	def get_dir_path(self, *dirs):
		"""Returns OS-independent path to directories given and creates resulting directory if not existing"""
		assert len(dirs) > 0
		# append all directory names
		path = os.path.join(dirs[0], *dirs[1:]) if len(dirs) > 1 else dirs[0]
		# make sure directory exists
		if not os.path.exists(path):
			os.makedirs(path)
		return path

	def get_file_path(self, dir_path, file_name):
		return os.path.join(dir_path, file_name)

	def open_file(self, path):
		_file = open(path, 'w')
		self.open_files[path] = _file
		return _file

	def get_file(self, path):
		if path in self.open_files:
			return self.open_files[path]
		else:
			return self.open_file(path)

	def log_line(self, message, _file, with_timestamp=False):
		line = ""
		if with_timestamp:
			timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
			line += timestamp + " - "
		line += message + "\n"
		_file.write(line)

	def flush_files(self):
		for _, _file in self.open_files.items():
			_file.flush()

	def close_files(self):
		for _, _file in self.open_files.items():
			_file.close()
		self.open_files.clear()
