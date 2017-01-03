import keras.optimizers
from keras.engine import Merge
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential

import sft.eps.Linear
import sft.agent.DeepQAgentReplayCloning
import sft.agent.model.KerasMlpModelNew
import sft.reward.TargetMiddle
from sft.log.AgentLogger import AgentLogger
from sft.config.complex import world
from sft.config.complex.world import *

logger = AgentLogger(__name__)

epsilon_update = sft.eps.Linear.Linear(
	start=1,
	end=0.1,
	steps=epochs * 3/5
)

optimizer = keras.optimizers.RMSprop(
	lr=0.00025  # learning rate
)

# just flatten view and action and then merge
model_view = Sequential()
model_view.add(Flatten(input_shape=(1, view_size.w, view_size.h)))
model_actions = Sequential()
model_actions.add(Flatten(input_shape=(1, action_hist_len, nb_actions)))
model = sft.agent.model.KerasMlpModelNew.KerasMlpModelNew(
	logger=logger,
	layers=[
		Merge([model_view, model_actions], mode='concat', concat_axis=1),
		Dense(16, init= 'lecun_uniform', activation='relu'),
		Dense(nb_actions, init='lecun_uniform', activation='linear')
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
	steps_clone=25
)

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
