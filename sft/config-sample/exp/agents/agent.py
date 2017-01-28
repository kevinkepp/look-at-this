from lasagne.layers import DenseLayer, FlattenLayer
from lasagne.nonlinearities import linear, rectify
from lasagne.updates import sgd

import sft.agent.DeepQAgentGpu
import sft.agent.model.LasagneMlpModel
import sft.eps.Linear
import sft.reward.TargetMiddle
from sft.agent.ah.No import No
from sft.log.AgentLogger import AgentLogger
from .. import world
from ..world import *

logger = AgentLogger(__name__)

epsilon_update = sft.eps.Linear.Linear(
	start=1,
	end=0.1,
	steps=epochs * 4 / 5
)

# use no action history
action_hist = No()
# use last actions as action history
"""action_hist = LastN(logger=logger, n=4)"""
# use sum of all actions as action history
"""action_hist = Sum(logger=logger)"""
# use running average over all actions as action history
"""action_hist = RunningAvg(logger=logger, n=4, factor=0.5)"""

def build_network(net_in_views, net_in_actions):
	net_views_out = FlattenLayer(net_in_views)
	# net_actions_out = FlattenLayer(net_in_actions)
	# net_concat = ConcatLayer([net_views_out, net_actions_out])
	# net_hid = DenseLayer(net_concat, num_units=64, nonlinearity=rectify)
	net_hid = DenseLayer(net_views_out, num_units=64, nonlinearity=rectify)
	net_out = DenseLayer(net_hid, num_units=4, nonlinearity=linear)
	return net_out

def optimizer(loss, params):
	# use sgd optimization
	opt = sgd(loss, params, learning_rate=0.01)
	# use rmsprop optimization
	"""opt = rmsprop(loss, params, learning_rate=0.001, rho=0.9, epsilon=1e-8)"""
	# use deepmind rmsprop optimization
	"""opt = deepmind_rmsprop(loss, params, learning_rate=0.00025, rho=0.95, epsilon=1e-2)"""
	return opt

batch_size = 32

model = sft.agent.model.LasagneMlpModel.LasagneMlpModel(
	logger=logger,
	batch_size=batch_size,
	discount=0.95,
	actions=actions,
	view_size=world.view_size,
	action_hist_size=action_hist.get_size(),
	network_builder=build_network,
	optimizer=optimizer,
	clone_interval=250,
	clip_delta=0  # 0 means no error clipping
)

agent = sft.agent.DeepQAgentGpu.DeepQAgentGpu(
	logger=logger,
	actions=actions,
	batch_size=batch_size,
	buffer_size=100000,
	start_learn=1000,
	learn_interval=2,
	view_size=world.view_size,
	action_hist=action_hist,
	model=model
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
