import sft.sampler.SinglePoint
import sft.sim.PathWorldLoader
from sft import Size, Point
from sft.Actions import Actions
from sft.log.WorldLogger import WorldLogger

epochs = 2000
world_size = Size(70, 70)
view_size = Size(7, 7)
actions = Actions.all
nb_actions = len(actions)
max_steps = 500
model_persist_steps = 200

world_logger = WorldLogger(__name__)

# use fixed coordinates as starting point
sampler = sft.sampler.SinglePoint.SinglePoint(Point(32, 19))
# use freely sampled point
"""sampler = sft.sampler.Simple.Simple()"""
# use expansive sampling of starting positions
"""sampler = sft.sampler.Expansive.Expansive(
	logger=world_logger,
	epochs_until_max=epochs * 1/3,
	min_sample_size=view_size * 1.5
)"""

# use given world images which are loaded from disk
world_gen = sft.sim.PathWorldLoader.PathWorldLoader(
	logger=world_logger,
	world_path="tmp/line-worldstates",
	view_size=view_size,
	world_size=world_size,
	sampler=sampler,
	path_in_init_view=True,  # True: enforce path in initial view
	target_not_in_init_view=True  # True: enforce target not in initial view
)
# use worlds with simple generated paths, i.e. paths consisting of just a straight line
"""world_gen = sft.sim.SimplePathWorldGenerator.SimplePathWorldGenerator(...)"""
# use worlds with generated paths that range from straight lines to paths with multiple corners
"""world_gen = sft.sim.PathWorldGenerator.PathWorldGenerator(...)"""
