from sft.sim.PathWorldGenerator import PathWorldGenerator


class SimplePathWorldGenerator(PathWorldGenerator):

	def __init__(self, logger, view_size, world_size, sampler, path_in_init_view=False,
				 target_not_in_init_view=False):
		# enforce simple paths consisting of one step, i.e. straight lines
		super(SimplePathWorldGenerator, self).__init__(logger, view_size, world_size, sampler,
													   path_length_min=1, path_length_max=1,
													   path_step_length_min=max(world_size.tuple()) / 3,
													   path_in_init_view=path_in_init_view,
													   target_not_in_init_view=target_not_in_init_view)
