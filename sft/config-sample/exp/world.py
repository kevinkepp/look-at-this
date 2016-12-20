import sft.log.WorldLogger
import sft.sampler.SamplerExpansive
import sft.sim.PathWorldGenerator
from sft.Actions import Actions
from sft import Size
import sft.log.WorldLogger

epochs = 2000
world_size = Size(50, 50)
view_size = Size(7, 7)
actions = Actions.all
nb_actions = len(actions)
max_steps = 500
action_hist_len = 4

world_logger = sft.log.WorldLogger.WorldLogger(__file__, __name__)

sampler = sft.sampler.SamplerExpansive.SamplerExpansive(
	logger=world_logger,
	epochs_until_max=1000,
	min_sample_size=Size(int(view_size.w / 2), int(view_size.h / 2))
)

world_gen = sft.sim.PathWorldGenerator.PathWorldGenerator(
	logger=world_logger,
	view_size=view_size,
	world_size=world_size,
	sampler=sampler,
	path_in_init_view=True
)
