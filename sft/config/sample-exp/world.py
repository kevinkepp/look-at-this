import sft.sim.PathWorldGenerator
from Actions import Actions
from sft import Size

epochs = 1000
world_size = Size(50, 50)
view_size = Size(7, 7)
nb_actions = len(Actions.all)
max_steps = 500
action_hist_len = 4

world_gen = sft.sim.PathWorldGenerator.PathWorldGenerator(
	view_size=view_size,
	world_size=world_size,
	path_in_init_view=True
)
