from keras.models import Sequential
from keras.models import load_model

import numpy as np

from sft.State import State
from sft.agent.model.DeepQModel import DeepQModel


class KerasMlpModel(DeepQModel):
	def __init__(self, logger, layers, loss, optimizer):
		self.logger = logger
		self.layers = layers
		self.layers = layers
		self.loss = loss
		self.optimizer = optimizer
		self._build_model(layers, loss, optimizer)

	def _build_model(self, layers, loss, optimizer):
		model = Sequential()
		for layer in layers:
			model.add(layer)
		model.compile(loss=loss, optimizer=optimizer)
		self._model = model

	def predict_qs(self, state):
		x = state.flatten()
		assert x.shape[1] == self.layers[0].input_shape[1]
		p = self._model.predict(x, batch_size=1, verbose=0)
		return p

	def update_qs(self, states, targets):
		X = State.flatten_states(states)
		assert X.shape[1] == self.layers[0].input_shape[1]
		n_samples = X.shape[0]
		self._model.fit(X, targets, batch_size=n_samples, nb_epoch=1, verbose=0)

	def load(self, file_path):
		self.logger.log_message("Load model from {0}".format(file_path))
		return load_model(file_path)

	def save(self, file_path):
		self.logger.log_message("Save model to {0}".format(file_path))
		self._model.save(file_path)

	def copy_from(self, other):
		assert isinstance(other, KerasMlpModel)
		self._model.set_weights(other._model.get_weights())

	def clone(self):
		# rebuild model
		model_cloned = type(self)(self.logger, self.layers, self.loss, self.optimizer)
		# copy weights
		model_cloned.copy_from(self)
		return model_cloned

