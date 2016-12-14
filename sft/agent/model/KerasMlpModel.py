from keras.models import Sequential
from keras.models import load_model

from State import State
from sft.agent.model.DeepQModel import DeepQModel


class KerasMlpModel(DeepQModel):
	def __init__(self, logger, layers, loss, optimizer):
		self.logger = logger
		self._build_model(layers, loss, optimizer)

	def _build_model(self, layers, loss, optimizer):
		model = Sequential()
		for layer in layers:
			model.add(layer)
		model.compile(loss=loss, optimizer=optimizer)
		self._model = model

	def predict_qs(self, state):
		x = state.flatten()
		p = self._model.predict(x, batch_size=1, verbose=0)
		return p

	def update_qs(self, states, targets):
		states = State.flatten_states(states)
		n_samples = states.shape[0]
		self._model.fit(states, targets, batch_size=n_samples, nb_epoch=1, verbose=0)

	def load_from_file(self, file_path):
		self.logger.log_message("Load model from {0}".format(file_path))
		return load_model(file_path)

	def save_to_file(self, file_path):
		self.logger.log_message("Save model to {0}".format(file_path))
		self._model.save(file_path)
