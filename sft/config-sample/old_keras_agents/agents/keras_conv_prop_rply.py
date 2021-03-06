import keras.optimizers
from keras.engine import Merge
from keras.layers import Dense, Flatten, Convolution2D
from keras.models import Sequential
from keras.regularizers import l2

import sft.eps.Linear
import sft.agent.DeepQAgentPosPathReplay
import sft.agent.model.KerasMlpModelNew
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

regularization = 0.01

# convolution on view and flatten actions
model_view = Sequential()
model_view.add(Convolution2D(16, 3, 3, W_regularizer=l2(regularization), activation='relu', border_mode='same', input_shape=(1, view_size.w, view_size.h)))
model_view.add(Flatten())
model_actions = Sequential()
model_actions.add(Flatten(input_shape=(1, action_hist_len, nb_actions)))
model = sft.agent.model.KerasMlpModelNew.KerasMlpModelNew(
	logger=logger,
	layers=[
		Merge([model_view, model_actions], mode='concat', concat_axis=1),
		Dense(32, activation='relu', W_regularizer=l2(regularization)),
		Dense(nb_actions, activation='linear', W_regularizer=l2(regularization))
	],
	loss='mse',
	optimizer=optimizer
)

agent = sft.agent.DeepQAgentPosPathReplay.DeepQAgentPosPathReplay(
	logger=logger,
	actions=actions,
	model=model,
	discount=0.9,
	batch_size=16,
	buffer_size=100000,
	start_learn=50,
	steps_clone=25,
	portions=[0.05, 0.5, 0.45]
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
