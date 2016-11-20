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

	def _train(self, env, name):
		print("Training {0} with {1} epochs".format(name, self.epochs))
		return env.run(self.epochs, trainingmode=True)

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
		plot_names = []
		plot_results = []
		for env, name in zip(self.envs, self.env_names):
			t0 = time.time()
			res = self._train(env, name)
			te = time.time()
			res = (res[0], len(res[1]), res[2])
			print("Training took {0} sec".format(int(te-t0)))
			# visualize if enough data
			if visualize:
				x, scores = self._eval(res, window_size)
				print("Performance {0} last {1} epochs: {2}".format(name, window_size, np.round(scores[-1], 3)))
				plot_names.append(name)
				plot_results.append(res)
		if visualize:
			self.visualizer.plot_results(plot_names, plot_results, self.epochs, window_size, self.params)

	def _train_until(self, env, condition, window_size):
		results = []
		for i in range(self.epochs):
			visible = True if self.epochs > 10 and i % int(self.epochs / 10) < 3 else False
			env.use_special_sampling(i, self.epochs)
			res = env.run_epoch(i, visible=visible, trainingmode=True)
			res = (res[0], len(res[1]), res[2])
			results.append(res)
			if i >= window_size:
				scores_window = [self.calc_score(success, steps, best) for success, steps, best in results[-window_size:]]
				score = sum(scores_window) / float(window_size)
				if self.epochs > 20 and i % int(self.epochs / 10) == 0:
					print("Epoch {0}/{1}: {2}".format(i, self.epochs, np.round(score, 3)))
				if condition(score):
					break
		return score, results

	def run_until(self, condition, visualize=True):
		visualize = visualize and self.epochs >= self._eval_epoch_min
		plot_names = []
		plot_results = []
		window_size = self._eval_epoch_avg()
		for env, name in zip(self.envs, self.env_names):
			print("Training {0} with max {1} epochs until condition is met".format(name, self.epochs))
			t0 = time.time()
			score, results = self._train_until(env, condition, window_size)
			te = time.time()
			print("Training took {0} sec".format(int(te - t0)))
			if visualize:
				print("Performance {0} last {1} epochs: {2}".format(name, window_size, np.round(score, 3)))
				plot_names.append(name)
				plot_results.append(results)
		if visualize:
			self.visualizer.plot_results(plot_names, plot_results, self.epochs, window_size, self.params)
