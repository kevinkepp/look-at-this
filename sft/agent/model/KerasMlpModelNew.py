import numpy as np

from sft.agent.model.KerasMlpModel import KerasMlpModel


class KerasMlpModelNew(KerasMlpModel):
	def __init__(self, logger, layers, loss, optimizer):
		super(KerasMlpModelNew, self).__init__(logger, layers, loss, optimizer)

	def predict_qs(self, state):
		# assume one channel and one sample
		X = [np.array([[l]]) for l in state.to_list()]
		# [ (1,1,5,5), (1,1,4,4) ]
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
		# [ (n, 1, 5, 5), (n, 1, 4, 4) ]
		self._model.fit(X, targets, batch_size=n_samples, nb_epoch=1, verbose=0)
		# weights = self._model.get_weights()
		# means = [np.mean(np.abs(w)) for w in weights]
		# maxs = [np.max(np.abs(w)) for w in weights]
		# print "weights - maxs: %s, means: %s" % (str(maxs), str(means))
