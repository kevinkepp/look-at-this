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
				 network_input_view, network_input_actions, network_output, optimizer):
		self.logger = logger
		self.discount = discount
		self.actions = actions
		self.view_size = view_size
		self.action_hist_size = action_hist_size
		self.net_in_view = network_input_view
		self.net_in_actions = network_input_actions
		self.net_out = network_output
		self.optimizer = optimizer
		self.shared_batch = SharedBatch(batch_size, view_size, action_hist_size)
		self.shared_state = SharedState(view_size, action_hist_size)
		self.train_fn = None
		self.predict_fn = None
		self.build_model()

	def build_model(self):
		views = T.tensor4('views')
		action_hists = T.tensor4('action_hists')
		actions = T.icol('actions')
		next_views = T.tensor4('next_views')
		next_action_hists = T.tensor4('next_action_hists')
		rewards = T.col('rewards')
		terminals = T.icol('terminals')

		# build loss function based on q value predictions
		q_vals = lasagne.layers.get_output(self.net_out, {self.net_in_view: views, self.net_in_actions: action_hists})
		next_q_vals = lasagne.layers.get_output(self.net_out, {self.net_in_view: next_views, self.net_in_actions: next_action_hists})
		actionmask = T.eq(T.arange(4).reshape((1, -1)), actions.reshape((-1, 1))).astype(theano.config.floatX)
		terminals_float = terminals.astype(theano.config.floatX)
		target = rewards + \
				 (T.ones_like(terminals_float) - terminals_float) * \
				 self.discount * T.max(next_q_vals, axis=1, keepdims=True)
		output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
		diff = target - output
		loss = diff ** 2  # TODO error clipping
		loss = T.mean(loss)  # batch accumulator sum or mean

		# build train function
		params = lasagne.layers.helper.get_all_params(self.net_out)
		updates = self.optimizer(loss, params)
		train_givens = self.shared_batch.givens(views, action_hists, actions, next_views, next_action_hists, rewards, terminals)
		self.train_fn = theano.function([], [loss], updates=updates, givens=train_givens)

		# build predict function
		predict_givens = self.shared_state.givens(views, action_hists)
		self.predict_fn = theano.function([], q_vals[0], givens=predict_givens)

	def predict_qs(self, views, action_hists):
		self.shared_state.set(views, action_hists)
		return self.predict_fn()

	def update_qs(self, views, action_hists, actions, next_views, next_action_hists, rewards, terminals):
		self.shared_batch.set(views, action_hists, actions, next_views, next_action_hists, rewards, terminals)
		loss = self.train_fn()
		return np.sqrt(loss)

	def save(self, path):
		print("{0}.save not implemented".format(self.__class__.__name__))