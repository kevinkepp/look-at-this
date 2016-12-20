import keras.optimizers
from keras.layers import Dense, Activation

import sft.agent.DeepQAgentReplayCloning
import sft.agent.model.KerasMlpModel
import sft.reward.TargetMiddleReward
from sft.logging.Logger import Logger
from .. import world
from ..world import *

logger = Logger(__file__, __name__, world.world_logger.get_exp_log_path())

epsilon_update = sft.EpsilonUpdate.Linear(
	start=1,
	end=0.1,
	steps=epochs  # anneal epsilon over all epochs
)

optimizer = keras.optimizers.RMSprop(
	lr=0.00025  # learning rate
)

_model_input_size = view_size.w * view_size.h + action_hist_len * nb_actions
model = sft.agent.model.KerasMlpModel.KerasMlpModel(
	logger=logger,
	layers=[
		Dense(input_shape=(_model_input_size,), output_dim=16, init='lecun_uniform'),
		Activation('relu'),
		Dense(output_dim=nb_actions, init='lecun_uniform'),
		Activation('linear')
	],
	loss='mse',
	optimizer=optimizer
)

agent = sft.agent.DeepQAgentReplayCloning.DeepQAgentReplayCloning(
	logger=logger,
	actions=actions,
	model=model,
	discount=0.99,
	batch_size=16,
	buffer_size=100000,
	start_learn=50,
	steps_clone=25
)

reward = sft.reward.TargetMiddleReward.TargetMiddleReward(
	reward_target=100,
	reward_not_target=0
)

# reward = sft.reward.SimpleReward.RewardAtTheEndForOneInTheMiddle()
