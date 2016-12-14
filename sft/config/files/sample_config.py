from Actions import Actions
from sft import Size
import sft.sim.PathSimulator
import sft.EpsilonUpdate
import keras.optimizers
import sft.agent.model.KerasMlpModel
import sft.agent.DeepQAgentReplay
import sft.reward.SimpleReward

epochs = 2000
nb_actions = len(Actions.all)
max_steps = 1500
action_hist_len = 4

simulator_params = dict(
	view_size=Size(15, 15),
	world_size=Size(150, 150),
	path_in_init_view=True
)
simulator = sft.sim.PathSimulator.PathSimulator(**simulator_params)

epsilon_update_params = dict(
	start=1,
	end=0.1,
	steps=epochs
)
epsilon_update = sft.EpsilonUpdate.Linear(**epsilon_update_params)

optimizer_params = dict(
	lr=0.00025  # learning rate
)
optimizer = keras.optimizers.RMSprop(**optimizer_params)

model_params = dict(
	# input size is number of values in view and action history (4 per action)
	l_in_size=simulator.view_size.w * simulator.view_size.h + action_hist_len * 4,
	l_hid_sizes=[16],
	l_out_size=nb_actions,
	loss='mse',
	optimizer=optimizer
)
model = sft.agent.model.KerasMlpModel.KerasMlpModel(**model_params)

agent_params = dict(
	actions=Actions.all,
	model=model,
	discount=0.99,
	batch_size=8,
	buffer_size=100
)
agent = sft.agent.DeepQAgentReplay.DeepQAgentReplay(**agent_params)

reward = sft.reward.SimpleReward.RewardAtTheEndForOneInTheMiddle()
