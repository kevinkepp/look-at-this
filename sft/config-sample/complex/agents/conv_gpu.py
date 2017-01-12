import sft.agent.DeepQAgentGpu
import sft.agent.model.LasagneMlpModel
import sft.eps.Linear
from lasagne.nonlinearities import linear, rectify
from lasagne.init import GlorotUniform
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer
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

l_in = InputLayer(shape=(None, 1, world.view_size.w, world.view_size.h))
l_hid = Conv2DLayer(l_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, W=GlorotUniform())
l_hid2 = Conv2DLayer(l_in, num_filters=32, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, W=GlorotUniform())
l_hid3 = DenseLayer(l_hid2, num_units=128, nonlinearity=rectify)
l_out = DenseLayer(l_hid3, num_units=4, nonlinearity=linear)

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
	network=l_out,
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
	model=model
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
