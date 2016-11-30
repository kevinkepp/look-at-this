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
		score = stps - bst
		score_fail = score
		return score if succ == 1 else score_fail

	def __init__(self, envs, env_names, epochs, visualizer, **params):
		self.envs = envs
		self.env_names = env_names
		self.epochs = epochs
		self.params = params
		self.visualizer = visualizer

	def _eval_epoch_avg(self):
		return int(self.epochs / 10)

	def _train_until(self, env, break_condition, window_size, visualize):
		results = []
		for i in range(self.epochs):
			visualize_epoch = visualize and self.epochs > 10 and i % int(self.epochs / 10) < 3
			# env.use_special_sampling(i, self.epochs)
			res = env.run(i, visualize=visualize_epoch, trainingmode=True)
			res = (res[0], len(res[1]), res[2])
			results.append(res)
			if i >= window_size:
				scores_window = [self.calc_score(success, steps, best) for success, steps, best in results[-window_size:]]
				score = sum(scores_window) / float(window_size)
				if self.epochs > 20 and i % int(self.epochs / 10) == 0:
					print("Epoch {0}/{1}: {2}".format(i, self.epochs, np.round(score, 3)))
				if break_condition is not None and break_condition(score):
					break
		return score, results

	def run(self, break_condition=None, visualize=True):
		# only visualize if enough epochs given
		visualize = visualize and self.epochs >= self._eval_epoch_min
		plot_names = []
		plot_results = []
		window_size = self._eval_epoch_avg()
		for env, name in zip(self.envs, self.env_names):
			print("Training {0} with max {1} epochs until condition is met".format(name, self.epochs))
			paras = "Parameters: epochs={0}".format(self.epochs)
			for n, v in sorted(self.params.items()):
				paras += ", {0}={1}".format(n, v)
			print(paras)
			# TODO: better integration then using constant string name of class
			if env.__class__.__name__ == "PathSimSimpleExpansiveSampler":
				print("Restarting Expansive Sampling")
				env.restartExpansiveSampling(self.epochs)
			t0 = time.time()
			score, results = self._train_until(env, break_condition, window_size, visualize)
			te = time.time()
			print("Training took {0} sec".format(int(te - t0)))
			if visualize:
				print("Performance {0} last {1} epochs: {2}".format(name, window_size, np.round(score, 3)))
				plot_names.append(name)
				plot_results.append(results)
		if visualize:
			self.visualizer.plot_results(plot_names, plot_results, self.epochs, window_size, self.params)
