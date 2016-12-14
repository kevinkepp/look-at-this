import keras.optimizers
from keras.layers import Dense, Activation

import sft.agent.DeepQAgentReplay
import sft.agent.model.KerasMlpModel
import sft.reward.SimpleReward
import sft.EpsilonUpdate

from sft.config.exp1.world import *

epsilon_update = sft.EpsilonUpdate.Linear(
	start=1,
	end=0.1,
	steps=epochs
)

optimizer = keras.optimizers.RMSprop(
	lr=0.00025  # learning rate
)

_model_input_size = view_size.w * view_size.h + action_hist_len * nb_actions
model = sft.agent.model.KerasMlpModel.KerasMlpModel(
	layers=[
		Dense(input_shape=(_model_input_size,), output_dim=nb_actions, init='lecun_uniform'),
		Activation('linear')
	],
	loss='mse',
	optimizer=optimizer
)

agent = sft.agent.DeepQAgentReplay.DeepQAgentReplay(
	actions=Actions.all,
	model=model,
	discount=0.99,
	batch_size=16,
	buffer_size=100
)

reward = sft.reward.SimpleReward.RewardAtTheEndForOneInTheMiddle()
