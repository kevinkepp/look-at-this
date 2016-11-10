from lat.DeepQModel import DeepQModel
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD


class KerasMlpModel(DeepQModel):
	def __init__(self, l_in_size, l_hid_sizes, l_out_size, loss='mse', optimizer=SGD()):
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

	def predict_qs(self, state):
		assert state.size == self._l_in_size
		x = state.reshape(1, self._l_in_size)
		p = self._model.predict(x, batch_size=1)
		return p

	def update_qs(self, state, target_qs):
		assert state.size == self._l_in_size
		assert target_qs.size == self._l_out_size
		x = state.reshape(1, self._l_in_size)
		y = target_qs
		self._model.fit(x, y, batch_size=1, nb_epoch=1, verbose=0)

	def load_weights(self, filepath):
		self._model.load_weights(filepath)

	def save_weights(self, filepath):
		self._model.save_weights(filepath)
