from __future__ import division
import os
import matplotlib
if "DISPLAY" not in os.environ:
	matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

class Evaluator(object):
	_eval_epoch_min = 50

	def calc_score(self, succ, stps, bst):
		# score = bst / len(stps)
		# score_fail = 0
		score = len(stps) - bst
		score_fail = score
		return score if succ == 1 else score_fail

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
		# plt.ylabel("min steps / steps taken")
		plt.ylabel("steps taken - min steps")
		plt.grid(True)

	def _post_visualization(self, window_size):
		plt.xlim((window_size - 1, self.epochs + 1))
		# plt.ylim((-0.02, 1.02))  # make 0. and 1. performance visible
		plt.legend(loc='lower center')
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		params = "_epochs=%d" % self.epochs
		for n, v in sorted(self.params.items()):
			params += "_{0}={1}".format(n, v)
		filename = "tmp/plots/" + timestamp + params + ".png"
		plt.savefig(filename, bbox_inches='tight')
		# if "DISPLAY" in os.environ:
		#	plt.show()

	def _eval(self, res, window_size):
		scores = []
		# average over last results
		for i in range(window_size, len(res)):
			# calculate performance as ratio of minimum possible number of steps to succeed compared to  number of steps
			# the agent took, or 0 if it failed
			scores_window = [self.calc_score(success, steps, best) for success, steps, best in res[i - window_size:i]]
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

	def _train_until(self, env, name, condition, window_size):
		results = []
		scores = []
		for i in range(self.epochs):
			visible = True if self.epochs > 10 and i % int(self.epochs / 10) < 3 else False
			res = env.run_epoch(i, visible=visible, trainingmode=True)
			results.append(res)
			if i >= window_size:
				scores_window = [self.calc_score(success, steps, best) for success, steps, best in results[-window_size:]]
				score = sum(scores_window) / float(window_size)
				if self.epochs > 10 and i % int(self.epochs / 10) == 0:
					print("Epoch {0}/{1}: {2}".format(i, self.epochs, np.round(score, 3)))
				scores.append(score)
				if condition(score):
					break
		return np.arange(window_size, window_size + len(scores)), scores

	def run_until(self, condition, visualize=True):
		visualize = visualize and self.epochs >= self._eval_epoch_min
		plot_data = []
		window_size = self._eval_epoch_avg()
		for env, name in zip(self.envs, self.env_names):
			print("Training {0} with max {1} epochs until condition is met".format(name, self.epochs))
			t0 = time.time()
			x, scores = self._train_until(env, name, condition, window_size)
			te = time.time()
			print("Training took {0} sec".format(int(te - t0)))
			if visualize:
				print("Performance {0} last {1} epochs: {2}".format(name, window_size, np.round(scores[-1], 3)))
				plot_data.append((x, scores, name))
		if visualize:
			self._pre_visualization(window_size)
			for x, y, name in plot_data:
				plt.plot(x, y, label=name)
			self._post_visualization(window_size)
