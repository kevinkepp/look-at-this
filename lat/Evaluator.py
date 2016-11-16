from __future__ import division
import os
import matplotlib
if not "DISPLAY" in os.environ:
	matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

class Evaluator(object):
	_eval_epoch_min = 50

	def __init__(self, envs, env_names, epochs, **params):
		self.envs = envs
		self.env_names = env_names
		self.epochs = epochs
		self.params = params

	def _eval_epoch_avg(self):
		return int(self.epochs / 10)

	def _train(self, env, name):
		print("Training {0} with {1} epochs".format(name, self.epochs))
		return env.run(self.epochs, trainingmode=True)

	def _pre_visualization(self, window_size):
		title = "Training over {0} epochs (steps avg over last {1} epochs)".format(self.epochs, window_size)
		plt.title(title)
		plt.xlabel("epochs")
		plt.ylabel("min steps / steps taken")
		plt.grid(True)

	def _post_visualization(self, window_size):
		plt.xlim((window_size - 1, self.epochs + 1))
		plt.ylim((-0.02, 1.02))  # make 0. and 1. performance visible
		plt.legend(loc='lower center')
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		params = "_epochs=%d" % self.epochs
		for n, v in sorted(self.params.items()):
			params += "_{0}={1}".format(n, v)
		filename = "tmp/" + timestamp + params + ".png"
		plt.savefig(filename, bbox_inches='tight')
		if "DISPLAY" in os.environ:
			plt.show()

	def _eval(self, res, window_size):
		scores = []
		# average over last results
		for i in range(window_size, len(res)):
			# calculate performance as ratio of minimum possible number of steps to succeed compared to  number of steps
			# the agent took, or 0 if it failed
			calc_score = lambda success, steps, best: best / len(steps) if success == 1 else 0
			scores_window = [calc_score(success, steps, best) for success, steps, best in res[i - window_size:i]]
			score = sum(scores_window) / window_size
			scores.append(score)
		return np.arange(window_size, len(res)), scores

	def run(self, visualize=False):
		window_size = self._eval_epoch_avg()
		visualize = visualize and self.epochs >= self._eval_epoch_min
		plot_data = []
		for env, name in zip(self.envs, self.env_names):
			t0 = time.time()
			res = self._train(env, name)
			te = time.time()
			print("Training took {0} sec".format(int(te-t0)))
			# visualize if enough data
			if visualize:
				x, scores = self._eval(res, window_size)
				print("Performance {0} last {1} epochs: {2}".format(name, window_size, np.round(scores[-1], 3)))
				plot_data.append((x, scores, name))
		if visualize:
			self._pre_visualization(window_size)
			for x, y, name in plot_data:
				plt.plot(x, y, label=name)
			self._post_visualization(window_size)

	def _train_until(self, env, name, min_performance, window_size):
		results = []
		scores = []
		calc_score = lambda succ, stps, bst: bst / len(stps) if succ == 1 else 0
		for i in range(self.epochs):
			res = env.run_epoch(i, trainingmode=True)
			results.append(res)
			if i >= window_size:
				scores_window = [calc_score(success, steps, best) for success, steps, best in results[-window_size:]]
				score = sum(scores_window) / float(window_size)
				if self.epochs > 10 and i % int(self.epochs / 10) == 0:
					print("Epoch {0}/{1}: {2}".format(i, self.epochs, np.round(score, 3)))
				scores.append(score)
				if score >= min_performance:
					break
		return np.arange(window_size, window_size + len(scores)), scores

	def run_until(self, min_performance, visualize=True):
		visualize = visualize and self.epochs >= self._eval_epoch_min
		plot_data = []
		window_size = self._eval_epoch_avg()
		for env, name in zip(self.envs, self.env_names):
			print("Training {0} with max {1} epochs until performance is {2}".format(name, self.epochs, min_performance))
			x, scores = self._train_until(env, name, min_performance, window_size)
			if visualize:
				print("Performance {0} last {1} epochs: {2}".format(name, window_size, np.round(scores[-1], 3)))
				plot_data.append((x, scores, name))
		if visualize:
			self._pre_visualization(window_size)
			for x, y, name in plot_data:
				plt.plot(x, y, label=name)
			self._post_visualization(window_size)
