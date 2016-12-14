from Actions import Actions
from config.ConfigParser import ConfigItem
from sft import Size

epochs = 3000
nb_actions = len(Actions.all)

simulator = ConfigItem("sft.sim.PathSimulator.PathSimulator")
simulator.view_size = Size(15, 15)
simulator.world_size = Size(150, 150)
simulator.path_in_init_view = True

epsilon_update = ConfigItem("sft.EpsilonUpdate.Linear")
epsilon_update.start = 1
epsilon_update.end = 0.1
epsilon_update.steps = epochs

optimizer = ConfigItem("keras.optimizers.RMSprop")
optimizer.lr = 0.00025  # learning rate

model = ConfigItem("sft.agent.model.KerasMlpModel.KerasMlpModel")
model.l_in_size = simulator.view_size.w * simulator.view_size.h
model.l_hid_sizes = [16]
model.l_out_size = nb_actions
model.loss = 'mse'
model.optimizer = optimizer

agent = ConfigItem("sft.agent.DeepQAgentReplay.DeepQAgentReplay")
agent.actions = Actions.all
agent.model = model
agent.discount = 0.99
agent.batch_size = 8
agent.buffer_size = 100
