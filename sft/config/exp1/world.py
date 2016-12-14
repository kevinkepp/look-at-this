import sft.sim.PathWorldGenerator
from Actions import Actions
from sft import Size

epochs = 2000
world_size = Size(150, 150)
view_size = Size(15, 15)
nb_actions = len(Actions.all)
max_steps = 1500
action_hist_len = 4

world_gen = sft.sim.PathWorldGenerator.PathWorldGenerator(
	view_size=view_size,
	world_size=world_size,
	path_in_init_view=True
)
