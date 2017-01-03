import numpy as np

from sft.agent.model.KerasMlpModel import KerasMlpModel


class KerasConvModel(KerasMlpModel):
	def __init__(self, logger, layers, loss, optimizer):
		super(KerasConvModel, self).__init__(logger, layers, loss, optimizer)

	def predict_qs(self, state):
		X = np.array([[state.view]])
		# shape (1, 1, view_size.w, view_size.h)
		s = (1, 1) + self.layers[0].input_shape[2:4]
		assert X.shape == s
		p = self._model.predict(X, batch_size=1, verbose=0)
		return p

	def update_qs(self, states, targets):
		n_samples = len(states)
		X = np.array([[s.view] for s in states])
		# shape (n_sample, 1, view_size.w, view_size.h)
		s = (n_samples, 1) + self.layers[0].input_shape[2:4]
		assert X.shape == s
		self._model.fit(X, targets, batch_size=n_samples, nb_epoch=1, verbose=0)
