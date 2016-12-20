from eps.Update import Update


class Constant(Update):
	def __init__(self, value):
		self.value = value

	def get_value(self, epoch):
		return self.value


