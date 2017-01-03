import keras.optimizers
from keras.layers import Dense

import sft.eps.Linear
import sft.agent.DeepQAgentReplayCloning
import sft.agent.model.KerasMlpModel
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

optimizer = keras.optimizers.RMSprop(
	lr=0.00025  # learning rate
)

_model_input_size = view_size.w * view_size.h + action_hist_len * nb_actions
model = sft.agent.model.KerasMlpModel.KerasMlpModel(
	logger=logger,
	layers=[
		Dense(input_shape=(_model_input_size,), output_dim=16, init='lecun_uniform', activation='relu'),
		Dense(output_dim=nb_actions, init='lecun_uniform', activation='linear')
	],
	loss='mse',
	optimizer=optimizer
)

agent = sft.agent.DeepQAgentReplayCloning.DeepQAgentReplayCloning(
	logger=logger,
	actions=actions,
	model=model,
	discount=0.9,
	batch_size=16,
	buffer_size=10000,
	start_learn=50,
	steps_clone=25
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
