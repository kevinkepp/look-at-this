from lasagne.layers import DenseLayer, ConcatLayer, FlattenLayer
from lasagne.nonlinearities import linear, rectify
from lasagne.updates import rmsprop

import sft.agent.DeepQAgentGpu
import sft.agent.model.LasagneMlpModel
import sft.eps.Linear
import sft.reward.TargetMiddle
from sft.log.AgentLogger import AgentLogger
from .. import world
from ..world import *

logger = AgentLogger(__name__)

action_hist_len = 4
action_hist_size = Size(action_hist_len, world.nb_actions)

epsilon_update = sft.eps.Linear.Linear(
	start=1,
	end=0.1,
	steps=epochs * 4/5
)

def build_network(net_in_views, net_in_actions):
	net_views_out = FlattenLayer(net_in_views)
	net_actions_out = FlattenLayer(net_in_actions)
	net_concat = ConcatLayer([net_views_out, net_actions_out])
	net_hid = DenseLayer(net_concat, num_units=16, nonlinearity=rectify)
	net_out = DenseLayer(net_hid, num_units=4, nonlinearity=linear)
	return net_out

def optimizer(loss, params):
	return rmsprop(loss, params,
		learning_rate=0.00025,
		rho=0.9,
		epsilon=1e-8
	)

batch_size = 32

model = sft.agent.model.LasagneMlpModel.LasagneMlpModel(
	logger=logger,
	batch_size=batch_size,
	discount=0.99,
	actions=actions,
	view_size=world.view_size,
	action_hist_size=action_hist_size,
	network_builder=build_network,
	optimizer=optimizer,
	clone_interval=500
)

agent = sft.agent.DeepQAgentGpu.DeepQAgentGpu(
	logger=logger,
	actions=actions,
	batch_size=batch_size,
	buffer_size=100000,
	start_learn=1000,
	learn_interval=1,
	view_size=world.view_size,
	action_hist_size=action_hist_size,
	model=model
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
