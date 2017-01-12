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
		# assume one channel and one sample
		X = [np.array([[l]]) for l in state.to_list()]
		# X shapes: [ (1, 1, view_size.w, view_size.h), (1, 1, action_history_len, nb_actions) ]
		p = self._model.predict(X, batch_size=1, verbose=0)
		return p

	def update_qs(self, states, targets):
		n_samples = len(states)
		assert len(states) > 0
		list_length = len(states[0].to_list())
		X = []
		for i in range(list_length):
			X.append([])
		for state in states:
			for i, v in enumerate(state.to_list()):
				v = [v]  # assume one channel
				X[i].append(v)
		for i, v in enumerate(X):
			X[i] = np.array(v)
		# X shapes: [ (n_samples, 1, view_size.w, view_size.h), (n_samples, 1, action_history_len, nb_actions) ]
		self._model.fit(X, targets, batch_size=n_samples, nb_epoch=1, verbose=0)
		# weights = self._model.get_weights()
		# means = [np.mean(np.abs(w)) for w in weights]
		# maxs = [np.max(np.abs(w)) for w in weights]
		# print "weights - maxs: %s, means: %s" % (str(maxs), str(means))

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

