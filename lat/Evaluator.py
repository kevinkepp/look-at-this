import matplotlib.pyplot as plt
import numpy as np


class Evaluator(object):
	_eval_epoch_min = 50

	def __init__(self, envs, env_names, epochs, **params):
		self._envs = envs
		self._env_names = env_names
		self._epochs = epochs
		self._params = params

	def _eval_epoch_avg(self):
		return int(self._epochs / 10)

	def _train(self, env, name):
		print("Training {0} with {1} epochs".format(name, self._epochs))
		return env.run(self._epochs, trainingmode=True)

	def _pre_visualization(self, window_size):
		title = "Training over {0} epochs (avg over last {1} epochs".format(self._epochs, window_size)
		for n, v in self._params.items():
			title += ", {0}: {1}".format(n, v)
		title += ")"
		plt.title(title)
		plt.xlabel("epochs")
		plt.ylabel("% correct")
		plt.grid(True)

	def _post_visualization(self, window_size):
		plt.xlim((window_size, self._epochs + 1))
		plt.ylim((0, 102))  # make 100% performance visible
		plt.legend(loc='lower center')
		plt.show()

	def _eval(self, res, name, window_size):
		scores = []
		# average over last results
		for i in range(window_size, len(res)):
			successes = [r[0] for r in res]
			steps = [r[1] for r in res]
			score = sum(successes[i - window_size:i]) / window_size * 100.
			scores.append(score)
		# visualize
		plt.plot(np.arange(window_size, len(res)), scores, label=name)
		print("Performance {0} last {1} epochs: {2}%".format(name, window_size, scores[-1]))

	def _eval_simple(self, res, name):
		print("Results {0}: {1}".format(name, res))

	# TODO visualize = True does not work because of plotting interference with Simulator
	# TODO fix by only storing results and plot at the end!
	def run(self, visualize=False):
		window_size = self._eval_epoch_avg()
		visualize = visualize and self._epochs >= self._eval_epoch_min
		if visualize:
			self._pre_visualization(window_size)
		for env, name in zip(self._envs, self._env_names):
			res = self._train(env, name)
			# visualize if enough data
			if visualize:
				self._eval(res, name, window_size)
			else:
				self._eval_simple(res, name)
		if visualize:
			self._post_visualization(window_size)
