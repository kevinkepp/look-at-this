from sft.sampler.Sampler import Sampler
from sft import sample_point_within, Point, Size, Rectangle


class SamplerExpansive(Sampler):

	def __init__(self, logger, epochs_until_max, min_sample_size):
		# self.logger = logger
		self.epoch = 0  # used to store current epoch
		# number of epochs at which we sample from max distance
		self.epochs_until_max = epochs_until_max
		self.min_sample_size = min_sample_size

	def sample_init_pos(self, bbox, target_pos):
		"""samples the initial position based on a bounding box and the targets position"""
		spl_box = self._get_sample_bbox(bbox, target_pos)
		self.epoch += 1
		return sample_point_within(spl_box)

	def _get_sample_bbox(self, bbox, target_pos):
		"""returns a new bbox from which to sample afterwards based on the current bounds and the target position"""
		# TODO incorporate self.min_sample_size
		h_spl_box = self._get_side_length(self.min_sample_size.h, bbox.h)
		w_spl_box = self._get_side_length(self.min_sample_size.w, bbox.w)
		x = target_pos.x - int(w_spl_box * 0.5)
		y = target_pos.y - int(h_spl_box * 0.5)
		pt_start_spl_box = Point(x, y)
		spl_box_size = Size(w_spl_box, h_spl_box)
		spl_box = Rectangle(pt_start_spl_box, spl_box_size)
		return spl_box.intersection(bbox)

	def _get_side_length(self, side_length_min, side_length_max):
		"""returns the length of the sample box for current episode"""
		if self.epoch <= self.epochs_until_max:
			d = side_length_max - side_length_min
			l = float(self.epoch) / self.epochs_until_max * d
			return l + side_length_min
		else:
			return side_length_max
