from sft.eval.Evaluator import Evaluator
import os

LOG_PATH = "tmp/logs"
EXPERIMENT = sorted(os.listdir(LOG_PATH))[-1]  # use newest directory by alphanumerical ordering
WORLD_DIR = "world"
AGENTS_DICT = {
	"Agent": "conv_gpu",
}

# results
SLIDING_MEAN_WINDOW = 100

# paths
PLOT_EVERY_KTH_EPOCH = 25
NUM_PLOT_PATHS_IN_ROW = 1

EXPERIMENT_PATH = os.path.join(LOG_PATH, EXPERIMENT)
print("Evaluating {0}".format(EXPERIMENT_PATH))
# create evaluator
evaluator = Evaluator(EXPERIMENT_PATH, WORLD_DIR, AGENTS_DICT)
print("Evaluate paths")
# plot results, paths, qs
evaluator.animate_epoch("Agent", 0, "q.tsv")
evaluator.plot_paths(PLOT_EVERY_KTH_EPOCH, NUM_PLOT_PATHS_IN_ROW, "q.tsv", text_every_kth=10)
print("Evaluate results")
evaluator.plot_results(AGENTS_DICT.keys(), SLIDING_MEAN_WINDOW)
print("Evaluate q-values")
evaluator.plot_qs(AGENTS_DICT.keys(), "q.tsv")
print("Evaluate epsilon-values")
evaluator.plot_one_value_parameter(AGENTS_DICT.keys(), "epsilon.tsv")
#evaluator.plot_expansive_spl_radius()
