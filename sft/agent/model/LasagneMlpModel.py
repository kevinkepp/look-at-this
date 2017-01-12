import numpy as np
from theano import *
import theano.tensor as T
import lasagne
import lasagne.layers


class SharedBatch(object):
	def __init__(self, size, view_size):
		self.size = size
		self.s = shared(np.zeros((self.size, 1, view_size.w, view_size.h), dtype=theano.config.floatX))
		self.a = shared(np.zeros((self.size, 1), dtype=np.int32), broadcastable=(False, True))
		self.s2 = shared(np.zeros((self.size, 1, view_size.w, view_size.h), dtype=theano.config.floatX))
		self.r = shared(np.zeros((self.size, 1), dtype=theano.config.floatX), broadcastable=(False, True))
		self.t = shared(np.zeros((self.size, 1), dtype=np.int32), broadcastable=(False, True))

	def set(self, s, a, s2, r, t):
		self.s.set_value(s)
		self.a.set_value(a)
		self.s2.set_value(s2)
		self.r.set_value(r)
		self.t.set_value(t)

	def givens(self, states, actions, next_states, rewards, terminals):
		return {
			states: self.s,
			actions: self.a,
			next_states: self.s2,
			rewards: self.r,
			terminals: self.t
		}


class SharedState(object):
	def __init__(self, view_size):
		self.view_size = view_size
		self.s = theano.shared(np.zeros((1, view_size.w, self.view_size.h), dtype=theano.config.floatX))

	def set(self, s):
		self.s.set_value(s)

	def givens(self, states):
		return {states: self.s.reshape((1, 1, self.view_size.w, self.view_size.h))}


class LasagneMlpModel(object):
	def __init__(self, logger, batch_size, discount, actions, learning_rate, view_size, network, optimizer):
		self.logger = logger
		self.discount = discount
		self.actions = actions
		self.learning_rate = learning_rate
		self.view_size = view_size
		self.network = network
		self.optimizer = optimizer
		self.shared_batch = SharedBatch(batch_size, view_size)
		self.shared_state = SharedState(view_size)
		self.train_fn = None
		self.predict_fn = None
		self.build_model()

	def build_model(self):
		states = T.tensor4('states')
		actions = T.icol('actions')
		next_states = T.tensor4('next_states')
		rewards = T.col('rewards')
		terminals = T.icol('terminals')

		# build loss function based on q value predictions
		q_vals = lasagne.layers.get_output(self.network, states)
		next_q_vals = lasagne.layers.get_output(self.network, next_states)
		actionmask = T.eq(T.arange(4).reshape((1, -1)), actions.reshape((-1, 1))).astype(theano.config.floatX)
		terminals_float = terminals.astype(theano.config.floatX)
		target = rewards + \
				 (T.ones_like(terminals_float) - terminals_float) * \
				 self.discount * T.max(next_q_vals, axis=1, keepdims=True)
		output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
		diff = target - output
		loss = 0.5 * diff ** 2  # no clipping
		loss = T.sum(loss)  # batch accumulator == 'sum'

		# build train function
		params = lasagne.layers.helper.get_all_params(self.network)
		updates = self.optimizer(loss, params)
		train_givens = self.shared_batch.givens(states, actions, next_states, rewards, terminals)
		self.train_fn = theano.function([], [loss], updates=updates, givens=train_givens)

		# build predict function
		predict_givens = self.shared_state.givens(states)
		self.predict_fn = theano.function([], q_vals[0], givens=predict_givens)

	def predict_qs(self, state):
		self.shared_state.set(state.reshape(1, self.view_size.w, self.view_size.h))
		return self.predict_fn()

	def update_qs(self, states, actions, next_states, rewards, terminals):
		self.shared_batch.set(states, actions, next_states, rewards, terminals)
		loss = self.train_fn()
		return np.sqrt(loss)

	def save(self, path):
		print("{0}.save not implemented".format(self.__class__.__name__))