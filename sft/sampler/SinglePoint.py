from sft.sampler.Sampler import Sampler


class SinglePoint(Sampler):

	def __init__(self, pt):
		self.pt = pt

	def sample_init_pos(self, bbox, target_pos):
		return self.pt

	def next_epoch(self):
		pass
