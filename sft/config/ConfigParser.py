import sys
import importlib
import inspect


class Config(object):
	@property
	def simulator(self):
		raise NotImplementedError

	@property
	def reward(self):
		raise NotImplementedError

	@property
	def agent(self):
		raise NotImplementedError


class ConfigItem(object):
	def __init__(self, class_name):
		self.clazz = class_name

	def get_arguments(self):
		d = self.__dict__
		return dict((k, d[k]) for k in d if k != "clazz")


class ConfigParser(object):
	def __init__(self, logger=None):
		self.logger = logger

	def parse(self, config_file):
		# TODO log (copy) config file
		config = Config()
		# import config file like a python module (which it is)
		config_file = self.import_module(config_file)
		items = inspect.getmembers(config_file)
		items = self.order_items(list(items))
		# parse global variables and config items which need to be instantiated
		self.parse_items(config, items)

	def order_items(self, names, res=[]):
		# TODO DFS to find possible build order
		return list(names)

	def parse_items(self, config, items):
		for name, value in items:
			if isinstance(value, ConfigItem):
				self.parse_config_item(config, name, value)
			elif not isinstance(value, object):
				self.parse_global_var(config, name, value)

	def parse_config_item(self, config, name, item):
		path = item.clazz.split(".")
		module = ".".join(path[:-1])
		class_name = path[-1]
		clazz = self.load_class(config, module, class_name)
		# instantiate class using all arguments given in config file
		args = item.get_arguments()
		self.parse_arguments(config, args)
		instance = clazz(**args)
		# store instance using name declared in config file
		config.name = instance

	def parse_global_var(self, config, name, value):
		setattr(config, name, value)

	def parse_arguments(self, config, args):
		for name in args:
			if hasattr(config, name):
				args[name] = config.name

	def load_class(self, config, module_name, class_name):
		try:
			module = self.import_module(module_name)
			return getattr(module, class_name)
		except ImportError as ie:
			# if module could not be loaded check if it's a reference to a config item
			if hasattr(config, module_name):
				return getattr(config, module_name)
			else:
				raise ie

	@staticmethod
	def import_module(name):
		return importlib.import_module(name)
