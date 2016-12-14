import numpy as np

from sft import Point


def sample_uniform(low, high=None):
	return np.random.randint(low, high)


def sample_normal(mean, std, low=float("-inf"), high=float("inf")):
	while True:
		idx = int(np.random.normal(mean, std, 1))
		if low <= idx <= high:
			return idx


def sample_point_within(rect):
	x = sample_uniform(rect.x, rect.x + rect.w)
	y = sample_uniform(rect.y, rect.y + rect.h)
	return Point(x, y)
