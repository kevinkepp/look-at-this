import sft.sampler.Simple
import sft.sampler.Expansive
import sft.sim.PathWorldGenerator
from sft.Actions import Actions
from sft import Size
from sft.log.WorldLogger import WorldLogger

epochs = 2000
world_size = Size(49, 49)
view_size = Size(7, 7)
actions = Actions.all
nb_actions = len(actions)
max_steps = max(world_size.tuple()) * 15
action_hist_len = 4

world_logger = WorldLogger(__name__)

# sampler = sft.sampler.Simple.Simple()

sampler = sft.sampler.Expansive.Expansive(
	logger=world_logger,
	epochs_until_max=epochs / 2,
	min_sample_size=view_size
)

world_gen = sft.sim.PathWorldGenerator.PathWorldGenerator(
	logger=world_logger,
	view_size=view_size,
	world_size=world_size,
	sampler=sampler,
	path_in_init_view=True,
	# enforce simple paths consisting of one step, i.e. straight lines
	path_length_min=1,
	path_length_max=1,
	path_step_length_min=max(world_size.tuple()) / 3
)
