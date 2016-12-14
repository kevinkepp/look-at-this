from __future__ import division

import numpy as np
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model

from sft.agent.model.DeepQModel import DeepQModel


class KerasMlpModel(DeepQModel):
	def __init__(self, l_in_size, l_hid_sizes, l_out_size, loss='mse', optimizer=RMSprop()):
		self._l_in_size = l_in_size
		self._l_hid_sizes = l_hid_sizes
		self._l_out_size = l_out_size
		self._build_model(loss, optimizer)

	def _build_model(self, loss, optimizer):
		model = Sequential()
		has_hidden = self._l_hid_sizes is not None and self._l_hid_sizes != []
		if has_hidden:
			# add first hidden layer
			model.add(Dense(self._l_hid_sizes[0], init='lecun_uniform', input_shape=(self._l_in_size,)))
			model.add(Activation('relu'))
			# model.add(Dropout(0.2))
			# add other hidden layers (if given)
			for hid_size in self._l_hid_sizes[1:]:
				model.add(Dense(hid_size, init='lecun_uniform'))
				model.add(Activation('relu'))
			# model.add(Dropout(0.2))
		# add output layer
		if has_hidden:
			model.add(Dense(self._l_out_size, init='lecun_uniform'))
		else:
			model.add(Dense(self._l_out_size, init='lecun_uniform', input_shape=(self._l_in_size,)))
		# linear output so we can have range of real-valued outputs
		model.add(Activation('linear'))
		model.compile(loss=loss, optimizer=optimizer)
		self._model = model

	@staticmethod
	def _flatten_state(state):
		view = state.view.reshape((1, state.view.size))
		actions = state.action_hist.reshape((1, state.action_hist.size))
		return np.hstack((view, actions))

	@staticmethod
	def _flatten_states(states):
		states = [KerasMlpModel._flatten_state(s) for s in states]
		states = [s.reshape(s.size, ) for s in states]
		return np.matrix(states)

	def predict_qs(self, state):
		assert state.view.size + state.action_hist.size == self._l_in_size
		x = self._flatten_state(state)
		p = self._model.predict(x, batch_size=1, verbose=0)
		return p

	def update_qs(self, states, targets):
		states = KerasMlpModel._flatten_states(states)
		n_samples = states.shape[0]
		n_features = states.shape[1]
		n_outputs = targets.shape[1]
		assert n_features == self._l_in_size
		assert n_outputs == self._l_out_size
		self._model.fit(states, targets, batch_size=n_samples, nb_epoch=1, verbose=0)

	def load_from_file(self, filepath):
		return load_model(filepath)

	def save_to_file(self, filepath):
		self._model.save(filepath)
