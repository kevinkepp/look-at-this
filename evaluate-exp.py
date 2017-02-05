from sft.eval.Evaluator import Evaluator
import os
from sft.log.AgentLogger import AgentLogger

LOG_PATH = "tmp/logs"
EXPERIMENT = "20170201-162649_exp_rmsprop_proprpl"  # sorted(os.listdir(LOG_PATH))[-1]  # use newest directory by alphanumerical ordering
EXPERIMENT_PATH = os.path.join(LOG_PATH, EXPERIMENT)

if True:
	AGENTS_DICT = {}
	for fa in os.listdir(EXPERIMENT_PATH):
		if fa.startswith(AgentLogger.LOG_AGENT_PREFIX):
			agent_model_path = fa
			agent_model_name = fa.split(AgentLogger.LOG_AGENT_PREFIX)[1][1:]
			AGENTS_DICT[agent_model_name] = agent_model_path
else:
	AGENTS_DICT = {
		"No AH": "agent_1_no_ah",
		"Short AH": "agent_2_short_ah",
		"Sum AH": "agent_3_sum_ah",
		"Avg AH": "agent_4_avg_ah",
		"Avg AH hid": "agent_5_avg_ah_hid",
	}

# results
SLIDING_MEAN_WINDOW = 50

# paths
PLOT_EVERY_KTH_EPOCH = 50
NUM_PLOT_PATHS_IN_ROW = 3

# get proper folders
WORLD_CONFIG_PATH = os.path.join(EXPERIMENT_PATH, "world/world.py")
WORLDS_PATH = os.path.join(EXPERIMENT_PATH, "world/world_init_logs")

print("Evaluating {0}".format(EXPERIMENT_PATH))
# create evaluator
evaluator = Evaluator(EXPERIMENT_PATH, WORLD_CONFIG_PATH, WORLDS_PATH, AGENTS_DICT)
if False:
	print("Create path animation")
	# only possible if paths already extracted
	evaluator.animate_epoch("agent", 1975, "q.tsv")
else:
	print("Evaluate results")
	evaluator.plot_results(AGENTS_DICT.keys(), SLIDING_MEAN_WINDOW)
	print("Evaluate q-values")
	evaluator.plot_qs(AGENTS_DICT.keys(), "q.tsv")
	print("Evaluate epsilon-values")
	# evaluator.plot_one_value_parameter(AGENTS_DICT.keys(), "epsilon.tsv")
	#evaluator.plot_expansive_spl_radius()
	print("Evaluate paths")
	# plot results, paths, qs
	# evaluator.plot_paths(PLOT_EVERY_KTH_EPOCH, NUM_PLOT_PATHS_IN_ROW, "q.tsv", text_every_kth=10)
