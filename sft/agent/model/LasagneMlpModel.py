from __future__ import print_function
import numpy as np
from theano import *
import theano.tensor as T
import lasagne
import lasagne.layers

from lasagne.regularization import regularize_layer_params, l2

class SharedBatch(object):
	def __init__(self, size, view_size, action_hist_size):
		self.size = size
		self.has_action_hist = action_hist_size.w > 0
		self.v = shared(np.zeros((self.size, 1, view_size.w, view_size.h), dtype=theano.config.floatX))
		if self.has_action_hist:
			self.ah = shared(
				np.zeros((self.size, 1, action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX))
		self.a = shared(np.zeros((self.size, 1), dtype=np.int32), broadcastable=(False, True))
		self.v2 = shared(np.zeros((self.size, 1, view_size.w, view_size.h), dtype=theano.config.floatX))
		if self.has_action_hist:
			self.ah2 = shared(
				np.zeros((self.size, 1, action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX))
		self.r = shared(np.zeros((self.size, 1), dtype=theano.config.floatX), broadcastable=(False, True))
		self.t = shared(np.zeros((self.size, 1), dtype=np.int32), broadcastable=(False, True))

	def set(self, v, ah, a, v2, ah2, r, t):
		self.v.set_value(v)
		if self.has_action_hist:
			self.ah.set_value(ah)
		self.a.set_value(a)
		self.v2.set_value(v2)
		if self.has_action_hist:
			self.ah2.set_value(ah2)
		self.r.set_value(r)
		self.t.set_value(t)

	def givens(self, views, action_hists, actions, next_states, next_action_hists, rewards, terminals):
		return {
			views: self.v,
			action_hists: self.ah,
			actions: self.a,
			next_states: self.v2,
			next_action_hists: self.ah2,
			rewards: self.r,
			terminals: self.t
		} if self.has_action_hist else {
			views: self.v,
			actions: self.a,
			next_states: self.v2,
			rewards: self.r,
			terminals: self.t
		}


class SharedState(object):
	def __init__(self, view_size, action_hist_size):
		self.view_size = view_size
		self.action_hist_size = action_hist_size
		self.has_action_hist = self.action_hist_size.w > 0
		self.v = theano.shared(np.zeros((view_size.w, view_size.h), dtype=theano.config.floatX))
		if self.has_action_hist:
			self.ah = theano.shared(np.zeros((action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX))

	def set(self, v, ah):
		self.v.set_value(v)
		if self.has_action_hist:
			self.ah.set_value(ah)

	def givens(self, views, action_hists):
		return {
			views: self.v.reshape((1, 1, self.view_size.w, self.view_size.h)),
			action_hists: self.ah.reshape((1, 1, self.action_hist_size.w, self.action_hist_size.h))
		} if self.has_action_hist else {
			views: self.v.reshape((1, 1, self.view_size.w, self.view_size.h))
		}


class LasagneMlpModel(object):
	"""clone_interval: 0 or negative disables cloning"""

	def __init__(self, logger, batch_size, discount, actions, view_size, action_hist_size,
				 network_builder, optimizer, clone_interval=0, clip_delta=0, regularization=1e-4, double_q=False):
		self.logger = logger
		self.batch_size = batch_size
		self.discount = discount
		self.actions = actions
		self.view_size = view_size
		self.action_hist_size = action_hist_size
		self.network_builder = network_builder
		self.optimizer = optimizer
		self.clone_interval = clone_interval
		self.clip_delta = clip_delta
		self.shared_batch = SharedBatch(batch_size, view_size, action_hist_size)
		self.shared_state = SharedState(view_size, action_hist_size)
		self.net_online_out = None
		self.net_target_out = None
		self.train_fn = None
		self.predict_fn = None
		self.regularization = regularization
		self.double_q = double_q
		self.build_model()
		self.steps_clone = 0
		self.all_layers = None

	def build_network(self, network_builder, view_size, action_hist_size):
		net_in_views = lasagne.layers.InputLayer(name='views', shape=(None, 1, view_size.w, view_size.h))
		net_in_actions = lasagne.layers.InputLayer(name='action_hists',
												   shape=(None, 1, action_hist_size.w, action_hist_size.h)) \
			if action_hist_size.w > 0 else None
		net_out_and_layers = network_builder(net_in_views, net_in_actions)
		if isinstance(net_out_and_layers, tuple):
			net_out, layers = net_out_and_layers
		else:
			net_out = net_out_and_layers
			layers = None
		return net_in_views, net_in_actions, net_out, layers

	def build_model(self):
		views_curr = T.tensor4('views')
		action_hists_curr = T.tensor4('action_hists')
		actions = T.icol('actions')
		views_next = T.tensor4('next_views')
		action_hists_next = T.tensor4('next_action_hists')
		rewards = T.col('rewards')
		terminals = T.icol('terminals')

		# initialize network(s) for computing q-values
		net_online_in_view, net_online_in_action_hist, self.net_online_out, self.all_layers = \
			self.build_network(self.network_builder, self.view_size, self.action_hist_size)
		net_online_in_curr = {net_online_in_view: views_curr, net_online_in_action_hist: action_hists_curr} \
			if self.action_hist_size.w > 0 else {net_online_in_view: views_curr}
		q_vals_online_curr = lasagne.layers.get_output(self.net_online_out, net_online_in_curr)
		# for predictions we always use the q-values estimated by the online network on the current state
		q_vals_pred = q_vals_online_curr
		if self.clone_interval > 0:
			net_target_in_view, net_target_in_action_hist, self.net_target_out, _ = \
				self.build_network(self.network_builder, self.view_size, self.action_hist_size)
			self._clone()
			net_target_in_next = {net_target_in_view: views_next, net_target_in_action_hist: action_hists_next} \
				if self.action_hist_size.w > 0 else {net_target_in_view: views_next}
			# predict q-values for next state with target network
			q_vals_target_next = lasagne.layers.get_output(self.net_target_out, net_target_in_next)
			if self.double_q:
				# Double Q-Learning:
				# use online network to choose best action on next state (q_vals_target_argmax)...
				net_online_in_next = {net_online_in_view: views_next, net_online_in_action_hist: action_hists_next} \
					if self.action_hist_size.w > 0 else {net_online_in_view: views_next}
				q_vals_online_next = lasagne.layers.get_output(self.net_online_out, net_online_in_next)
				q_vals_target_argmax = T.argmax(q_vals_online_next, axis=1, keepdims=False)
				# ...but use target network to estimate q-values for these actions
				q_vals_target = T.diagonal(T.take(q_vals_target_next, q_vals_target_argmax, axis=1)).reshape((-1, 1))
			else:
				q_vals_target = T.max(q_vals_target_next, axis=1, keepdims=True)
		else:
			net_target_in_next = {net_online_in_view: views_next, net_online_in_action_hist: action_hists_next} \
				if self.action_hist_size.w > 0 else {net_online_in_view: views_next}
			q_vals_online_next = lasagne.layers.get_output(self.net_online_out, net_target_in_next)
			q_vals_target = T.max(q_vals_online_next, axis=1, keepdims=True)
		# define loss computation
		actionmask = T.eq(T.arange(len(self.actions)).reshape((1, -1)), actions.reshape((-1, 1))).astype(theano.config.floatX)
		terminals_float = terminals.astype(theano.config.floatX)
		target = rewards + \
				 (T.ones_like(terminals_float) - terminals_float) * \
				 self.discount * q_vals_target
		output = (q_vals_pred * actionmask).sum(axis=1).reshape((-1, 1))
		diff = target - output
		if self.clip_delta > 0:
			# see https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/q_network.py
			quadratic_part = T.minimum(abs(diff), self.clip_delta)
			linear_part = abs(diff) - quadratic_part
			loss = quadratic_part ** 2 + self.clip_delta * linear_part
		else:
			loss = diff ** 2

		# regularization
		if self.all_layers is not None and self.regularization > 0:
			l2reg = 0
			for lll in self.all_layers:
				l2reg += regularize_layer_params(lll, l2) * self.regularization
			loss = T.mean(loss) + l2reg  # batch accumulator sum or mean
		else:
			loss = T.mean(loss)

		# define network update for training
		params = lasagne.layers.helper.get_all_params(self.net_online_out)
		updates = self.optimizer(loss, params)
		train_givens = self.shared_batch.givens(views_curr, action_hists_curr, actions, views_next, action_hists_next, rewards,
												terminals)
		self.train_fn = theano.function([], [loss], updates=updates, givens=train_givens)

		# define output prediction
		predict_givens = self.shared_state.givens(views_curr, action_hists_curr)
		self.predict_fn = theano.function([], q_vals_pred[0], givens=predict_givens)

	def _clone(self):
		param_values = lasagne.layers.get_all_param_values(self.net_online_out)
		lasagne.layers.set_all_param_values(self.net_target_out, param_values)

	def predict_qs(self, view, action_hist):
		self.shared_state.set(view, action_hist)
		return self.predict_fn()

	def update_qs(self, views, action_hists, actions, next_views, next_action_hists, rewards, terminals):
		self.shared_batch.set(views, action_hists, actions, next_views, next_action_hists, rewards, terminals)
		loss = self.train_fn()
		self.logger.log_parameter("loss", loss)
		self.steps_clone += 1
		if self.steps_clone == self.clone_interval:
			self._clone()
			self.steps_clone = 0
		self.log_weights_diff()
		self.log_weights()
		return loss

	def save(self, file_path):
		self.logger.log_message("Save model to {0}".format(file_path))
		np.savez(file_path, *lasagne.layers.get_all_param_values(self.net_online_out))

	def load(self, file_path):
		self.logger.log_message("Load model from {0}".format(file_path))
		with np.load(file_path) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.net_online_out, param_values)

	def log_weights_diff(self):
		""" check if weights of model and model_cloned are actually different one step before cloning"""
		if self.steps_clone == self.clone_interval - 1:
			param_values_next = lasagne.layers.get_all_param_values(self.net_target_out)
			param_values_org = lasagne.layers.get_all_param_values(self.net_online_out)
			diffs = np.zeros(len(param_values_org), dtype=theano.config.floatX)
			for i in range(len(param_values_org)):
				diffs[i] = np.sum(param_values_next[i] - param_values_org[i])
			diff = np.sum(diffs)
			self.logger.log_parameter("weights_diff", diff)
			assert diff != 0

	def log_weights(self):
		layers = lasagne.layers.get_all_param_values(self.net_online_out)
		# calculate min, max, mean, std per layer
		for layer in range(len(layers)):
			weights = layers[layer]
			self.logger.log_parameter("weights",
									  [layer, weights.shape, np.min(weights), np.max(weights), np.mean(weights), np.std(weights)],
									  headers=["layer", "shape", "min", "max", "mean", "std"])


def deepmind_rmsprop(loss_or_grads, params, learning_rate, rho, epsilon):
	from lasagne.updates import get_or_compute_grads
	from collections import OrderedDict

	grads = get_or_compute_grads(loss_or_grads, params)
	updates = OrderedDict()
	for param, grad in zip(params, grads):
		value = param.get_value(borrow=True)
		acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype),
								 broadcastable=param.broadcastable)
		acc_grad_new = rho * acc_grad + (1 - rho) * grad
		acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype),
								broadcastable=param.broadcastable)
		acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2
		updates[acc_grad] = acc_grad_new
		updates[acc_rms] = acc_rms_new
		updates[param] = (param - learning_rate * (grad /
						   T.sqrt(acc_rms_new - acc_grad_new ** 2 + epsilon)))
	return updates
