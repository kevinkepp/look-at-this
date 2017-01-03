import keras.optimizers
from keras.layers import Dense, Flatten, Convolution2D

import sft.eps.Linear
import sft.agent.DeepQAgentReplayCloning
import sft.agent.model.KerasConvModel
import sft.reward.TargetMiddle
from sft.log.AgentLogger import AgentLogger
from .. import world
from ..world import *

logger = AgentLogger(__name__)

epsilon_update = sft.eps.Linear.Linear(
	start=1,
	end=0.1,
	steps=epochs * 3/5
)

optimizer = keras.optimizers.RMSprop(
	lr=0.00025  # learning rate
)

_model_input_size = view_size.w * view_size.h + action_hist_len * nb_actions
model = sft.agent.model.KerasConvModel.KerasConvModel(
	logger=logger,
	layers=[
		# ignore action history for now
		Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(1, view_size.w, view_size.h)),
		Flatten(),
		Dense(32, activation='relu'),
		Dense(nb_actions, activation='linear')
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
	buffer_size=100000,
	start_learn=50,
	steps_clone=20
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
