import numpy as np
import os
import matplotlib.pyplot as plt

def _read_tsv(path):
	_file = open(path)
	headline = _file.readline().split("\t")
	vals = []
	for line in _file:
		if line is not "":
			vals.append(line.split("\t"))
	return headline, vals

def _process_w(val):
	o = []
	for l in val:
		e = int(l[0])
		layer = int(l[1])
		shape = l[2]
		min = float(l[3])
		max = float(l[4])
		mn = float(l[5])
		std = float(l[6].strip("\n"))
		o.append([e, layer, min, max, mn, std])
	return np.array(o)

def _process_w_diff(val):
	o = []
	for l in val:
		e = int(l[0])
		w_d = float(l[1].strip("\n"))
		o.append([e, w_d])
	return np.array(o)


exp_path = "tmp/logs/20170205-230039_exp_expansive"
agent = "agent_propreplay_runah_reg6_1" # agent_propreplay_runah_reg6_1
param_logs = "parameter_logs"
w_file = "weights.tsv"
w_diff_file = "weights_diff.tsv"

path_w = os.path.join(exp_path, agent, param_logs, w_file)
path_w_diff = os.path.join(exp_path, agent, param_logs, w_diff_file)

path_tmp = os.path.join(exp_path, agent + "_tmp.npy")

if not os.path.exists(path_tmp):
	_,f = _read_tsv(path_w)
	w = _process_w(f)
	np.save(path_tmp, w)
else:
	w = np.load(path_tmp)

_,f = _read_tsv(path_w_diff)
wd = _process_w_diff(f)

print(w.shape)

in_layer_w = "(65,64)"  # layer nr 0
in_layer_b = "(64,)"  	# layer nr 1
out_layer_w = "(64,4)"  # layer nr 2
out_layer_b = "(4,)"  	# layer nr 3

# [e, layer, min, max, mn, std]
plt.hold(True)
y = w[:,1] == 1
print(y.shape)

x = np.arange(y.shape[0])

layer = [1,3]
lbl = ["w_in", "b_in", "w_out", "b_out"]
clr = ["r", "g", "b", "y"]

plt.title(agent)

for i in layer:
	y = w[w[:,1] == i]
	x = np.arange(y.shape[0])
	plt.fill_between(x, y[:,2], y[:,3], label=lbl[i], color=clr[i], alpha=0.2)

# plt.plot(w[w[:,1] == 0][:,3], label="w_in")
# plt.plot(w[w[:,1] == 2][:,3], label="w_out")
# plt.plot(w[w[:,1] == 1][:,3], label="b_in")
# plt.plot(w[w[:,1] == 3][:,3], label="b_out")


plt.legend(loc="upper left")
#plt.plot(wd[:,1])

plt.show()

