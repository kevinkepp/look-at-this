class DummyLogger(object):
	def next_epoch(self):
		pass

	def log_parameter(self, para_name, para_val, headers=None):
		pass

	def log_results(self, actions_taken, success):
		pass

	def log_model(self, model, name=None):
		pass

	def log_message(self, message):
		pass

	def flush_files(self):
		pass

	def close_files(self):
		pass

	def log_init_state_and_world(self, world_state, agent_pos):
		pass
