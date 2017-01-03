import sft.sampler.Expansive
import sft.sim.PathWorldGenerator
from sft.Actions import Actions
from sft import Size
from sft.log.WorldLogger import WorldLogger

epochs = 5000
world_size = Size(50, 50)
view_size = Size(5, 5)
actions = Actions.all
nb_actions = len(actions)
max_steps = max(world_size.tuple()) * 15
action_hist_len = 4

world_logger = WorldLogger(__name__)

# sampler = sft.sampler.Simple.Simple()

sampler = sft.sampler.Expansive.Expansive(
	logger=world_logger,
	epochs_until_max=epochs * 1/3,
	min_sample_size=view_size * 1.5
)

world_gen = sft.sim.PathWorldGenerator.PathWorldGenerator(
	logger=world_logger,
	view_size=view_size,
	world_size=world_size,
	sampler=sampler,
	path_in_init_view=True,  # True: enforce path in initial view
	target_not_in_init_view=True  # True: enforce target not in initial view
)
