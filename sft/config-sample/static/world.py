import sft.sampler.SinglePoint
import sft.sim.PathWorldLoader
from sft.Actions import Actions
from sft import Size, Point
from sft.log.WorldLogger import WorldLogger

epochs = 2000
world_size = Size(70, 70)
view_size = Size(7, 7)
actions = Actions.all
nb_actions = len(actions)
max_steps = max(world_size.tuple()) * 15

world_logger = WorldLogger(__name__)

sampler = sft.sampler.SinglePoint.SinglePoint(
	Point(32, 19)
)

world_gen = sft.sim.PathWorldLoader.PathWorldLoader(
	logger=world_logger,
	world_path="tmp/line-worldstates",
	view_size=view_size,
	world_size=world_size,
	sampler=sampler,
	path_in_init_view=True,  # True: enforce path in initial view
	target_not_in_init_view=True  # True: enforce target not in initial view
)
