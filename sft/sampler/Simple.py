from sft import sample_point_within
from sft.sampler.Sampler import Sampler


class Simple(Sampler):

	def sample_init_pos(self, bbox, target_pos):
		return sample_point_within(bbox)
