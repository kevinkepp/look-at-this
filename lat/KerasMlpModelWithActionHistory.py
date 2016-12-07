from lat.KerasMlpModel import KerasMlpModel


class KerasMlpModelWithActionHistory(KerasMlpModel):

	def __init__(self, l_in_size, l_hid_sizes, l_out_size, loss='mse', optimizer=RMSprop(), action_hist_len=4, num_actions=4):
		super(KerasMlpModelWithActionHistory,self).__init__(l_in_size, l_hid_sizes, l_out_size, loss='mse', optimizer=RMSprop())
		self._l_in_size += num_actions * action_hist_len
		self.action_hist_len = action_hist_len
