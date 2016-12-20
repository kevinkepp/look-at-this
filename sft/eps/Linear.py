from eps.Update import Update


class Linear(Update):
	"""Anneals linearly from 'start' to 'end' over 'steps' steps"""

	def __init__(self, start, end, steps):
		self.start = start
		self._step_size = (end - start) / steps

	def get_value(self, epoch):
		return self.start + self._step_size * epoch

# custom update function with convex shape, adapted -log(x) with f(1) = 1 and f(EPOCHS) = 0
# def EPSILON_UPDATE_KEV(n):
# a = (np.exp(-1) * EPOCHS - 1) / (1 - np.exp(-1))
# return max(-np.log((n + a) / (EPOCHS + a)), EPSILON_MIN)
