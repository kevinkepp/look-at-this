import sft.agent.DeepQAgentGpu
import sft.agent.model.LasagneMlpModel
import sft.eps.Linear
from lasagne.nonlinearities import linear, rectify
from lasagne.layers import InputLayer, DenseLayer, ConcatLayer, FlattenLayer
from lasagne.updates import rmsprop
import sft.reward.TargetMiddle
from sft.log.AgentLogger import AgentLogger
from .. import world
from ..world import *

logger = AgentLogger(__name__)

epsilon_update = sft.eps.Linear.Linear(
	start=1,
	end=0.1,
	steps=epochs * 4/5
)

action_hist_size = Size(world.action_hist_len, world.nb_actions)

net_views_in = InputLayer(name='views', shape=(None, 1, world.view_size.w, world.view_size.h))
net_views_out = FlattenLayer(net_views_in)
net_actions_in = InputLayer(name='action_hists', shape=(None, 1, action_hist_size.w, action_hist_size.h))
net_actions_out = FlattenLayer(net_actions_in)
net_concat = ConcatLayer([net_views_out, net_actions_out])
net_hid = DenseLayer(net_concat, num_units=16, nonlinearity=rectify)
net_out = DenseLayer(net_hid, num_units=4, nonlinearity=linear)

optimizer = lambda loss, params: rmsprop(loss, params,
	learning_rate=0.00025,
	rho=0.9,
	epsilon=1e-8)

batch_size = 16

model = sft.agent.model.LasagneMlpModel.LasagneMlpModel(
	logger=logger,
	batch_size=batch_size,
	discount=0.9,
	actions=actions,
	learning_rate=0.001,
	view_size=world.view_size,
	action_hist_size=action_hist_size,
	network_input_view=net_views_in,
	network_input_actions=net_actions_in,
	network_output=net_out,
	optimizer=optimizer
)

agent = sft.agent.DeepQAgentGpu.DeepQAgentGpu(
	logger=logger,
	actions=actions,
	batch_size=batch_size,
	buffer_size=10000,
	start_learn=50,
	learn_steps=1,
	view_size=world.view_size,
	action_hist_size=action_hist_size,
	model=model
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
