from sft.sampler.Sampler import Sampler
from sft import sample_point_within, Point, Size, Rectangle


class SamplerExpansive(Sampler):

	def __init__(self, logger, epochs, pct_epochs_until_max, min_sample_dist_w, min_sample_dist_h):
		self.epoch = 0  # used to store current epoch
		self.epochs = epochs
		self.pct_epochs_until_max = pct_epochs_until_max  # percentage of epochs at which you reach the max distance to sample from
		self.min_sample_dist_w = min_sample_dist_w
		self.min_sample_dist_h = min_sample_dist_h
		self.logger = logger

	def sample_init_pos(self, bbox, target_pos):
		"""samples the initial position based on a bounding box and the targets position"""
		spl_box = self._get_sample_bbox(bbox, target_pos)
		return sample_point_within(spl_box)

	def _get_sample_bbox(self, bbox, target_pos):
		"""returns a new bbox from which to sample afterwards based on the current bounds and the target position"""
		h_spl_box = self._get_side_length(bbox.h)
		w_spl_box = self._get_side_length(bbox.w)
		x = target_pos.x - int(w_spl_box * 0.5)
		y = target_pos.y - int(h_spl_box * 0.5)
		pt_start_spl_box = Point(x, y)
		spl_box_size = Size(w_spl_box, h_spl_box)
		spl_box = Rectangle(pt_start_spl_box, spl_box_size)
		return spl_box.intersection(bbox)

	def _get_side_length(self, side_length):
		"""returns the length of the sample box for current episode"""
		l = float(self.epoch) / (self.epochs * self.pct_epochs_until_max) * side_length

		return max(l, side_length)
