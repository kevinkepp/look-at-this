from __future__ import print_function
import numpy as np
from theano import *
import theano.tensor as T
import lasagne
import lasagne.layers


class SharedBatch(object):
	def __init__(self, size, view_size, action_history_size):
		self.size = size
		self.v = shared(np.zeros((self.size, 1, view_size.w, view_size.h), dtype=theano.config.floatX))
		self.ah = shared(np.zeros((self.size, 1, action_history_size.w, action_history_size.h), dtype=theano.config.floatX))
		self.a = shared(np.zeros((self.size, 1), dtype=np.int32), broadcastable=(False, True))
		self.v2 = shared(np.zeros((self.size, 1, view_size.w, view_size.h), dtype=theano.config.floatX))
		self.ah2 = shared(np.zeros((self.size, 1, action_history_size.w, action_history_size.h), dtype=theano.config.floatX))
		self.r = shared(np.zeros((self.size, 1), dtype=theano.config.floatX), broadcastable=(False, True))
		self.t = shared(np.zeros((self.size, 1), dtype=np.int32), broadcastable=(False, True))

	def set(self, v, ah, a, v2, ah2, r, t):
		self.v.set_value(v)
		self.ah.set_value(ah)
		self.a.set_value(a)
		self.v2.set_value(v2)
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
		}


class SharedState(object):
	def __init__(self, view_size, action_hist_size):
		self.view_size = view_size
		self.action_hist_size = action_hist_size
		self.v = theano.shared(np.zeros((view_size.w, view_size.h), dtype=theano.config.floatX))
		self.ah = theano.shared(np.zeros((action_hist_size.w, action_hist_size.h), dtype=theano.config.floatX))

	def set(self, v, ah):
		self.v.set_value(v)
		self.ah.set_value(ah)

	def givens(self, views, action_hists):
		return {
			views: self.v.reshape((1, 1, self.view_size.w, self.view_size.h)),
			action_hists: self.ah.reshape((1, 1, self.action_hist_size.w, self.action_hist_size.h))
		}


class LasagneMlpModel(object):
	def __init__(self, logger, batch_size, discount, actions, view_size, action_hist_size,
				 network_builder, optimizer, clone_interval=0):
		self.logger = logger
		self.batch_size = batch_size
		self.discount = discount
		self.actions = actions
		self.view_size = view_size
		self.action_hist_size = action_hist_size
		self.network_builder = network_builder
		self.optimizer = optimizer
		self.clone_interval = clone_interval
		self.shared_batch = SharedBatch(batch_size, view_size, action_hist_size)
		self.shared_state = SharedState(view_size, action_hist_size)
		self.net_out = None
		self.net_out_next = None
		self.train_fn = None
		self.predict_fn = None
		self.build_model()
		self.steps_clone = 0

	def build_network(self, network_builder, view_size, action_hist_size):
		net_in_views = lasagne.layers.InputLayer(name='views', shape=(None, 1, view_size.w, view_size.h))
		net_in_actions = lasagne.layers.InputLayer(name='action_hists',
											shape=(None, 1, action_hist_size.w, action_hist_size.h))
		net_out = network_builder(net_in_views, net_in_actions)
		return net_in_views, net_in_actions, net_out

	def build_model(self):
		views = T.tensor4('views')
		action_hists = T.tensor4('action_hists')
		actions = T.icol('actions')
		next_views = T.tensor4('next_views')
		next_action_hists = T.tensor4('next_action_hists')
		rewards = T.col('rewards')
		terminals = T.icol('terminals')

		# initialize network(s) for computing q-values
		net_in_view, net_in_actions, self.net_out = self.build_network(self.network_builder, self.view_size,
																	   self.action_hist_size)
		q_vals = lasagne.layers.get_output(self.net_out, {net_in_view: views, net_in_actions: action_hists})
		if self.clone_interval > 0:
			net_in_view_next, net_in_actions_next, self.net_out_next = self.build_network(self.network_builder,
																		self.view_size, self.action_hist_size)
			next_q_vals = lasagne.layers.get_output(self.net_out_next,
													{net_in_view_next: next_views, net_in_actions_next: next_action_hists})
			self._clone()
		else:
			next_q_vals = lasagne.layers.get_output(self.net_out,
													{net_in_view: next_views, net_in_actions: next_action_hists})
		# define loss computation
		actionmask = T.eq(T.arange(4).reshape((1, -1)), actions.reshape((-1, 1))).astype(theano.config.floatX)
		terminals_float = terminals.astype(theano.config.floatX)
		target = rewards + \
				 (T.ones_like(terminals_float) - terminals_float) * \
				 self.discount * T.max(next_q_vals, axis=1, keepdims=True)
		output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
		diff = target - output
		loss = diff ** 2  # TODO error clipping
		loss = T.mean(loss)  # batch accumulator sum or mean

		# define network update for training
		params = lasagne.layers.helper.get_all_params(self.net_out)
		updates = self.optimizer(loss, params)
		train_givens = self.shared_batch.givens(views, action_hists, actions, next_views, next_action_hists, rewards, terminals)
		self.train_fn = theano.function([], [loss], updates=updates, givens=train_givens)

		# define output prediction
		predict_givens = self.shared_state.givens(views, action_hists)
		self.predict_fn = theano.function([], q_vals[0], givens=predict_givens)

	def _clone(self):
		param_values = lasagne.layers.get_all_param_values(self.net_out)
		lasagne.layers.set_all_param_values(self.net_out_next, param_values)

	def predict_qs(self, views, action_hists):
		self.shared_state.set(views, action_hists)
		return self.predict_fn()

	def update_qs(self, views, action_hists, actions, next_views, next_action_hists, rewards, terminals):
		self.shared_batch.set(views, action_hists, actions, next_views, next_action_hists, rewards, terminals)
		loss = self.train_fn()
		self.steps_clone += 1
		if self.steps_clone == self.clone_interval:
			self._clone()
			self.steps_clone = 0
		""" check if weights of model and model_cloned are actually different one step before cloning
		if self.steps_clone == self.clone_interval - 1:
			param_values_next = lasagne.layers.get_all_param_values(self.net_out_next)
			param_values_org = lasagne.layers.get_all_param_values(self.net_out)
			diffs = np.zeros(len(param_values_org))
			for i in range(len(param_values_org)):
				diffs[i] = np.sum(param_values_next[i] - param_values_org[i])
			diff = np.sum(diffs)
			print(diff)
			assert diff != 0 """
		return loss

	def save(self, file_path):
		self.logger.log_message("Save model to {0}".format(file_path))
		np.savez(file_path, *lasagne.layers.get_all_param_values(self.net_out))

	def load(self, file_path):
		self.logger.log_message("Load model from {0}".format(file_path))
		with np.load(file_path) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.net_out, param_values)
