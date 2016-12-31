from sft.eval.Evaluator import Evaluator
import os

LOG_PATH = "/home/philibb/Dropbox/Uni/Robotics Project/github-repo/tmp/logs"
EXPERIMENT = "20161231-094648_exp1"
WORLD_DIR = "world"
AGENTS_DICT = {
	"Deep[16]": "agent0"}

# results
SLIDING_MEAN_WINDOW = 100

# paths
PLOT_EVERY_KTH_EPOCH = 3
NUM_PLOT_PATHS_IN_ROW = 1

EXPERIMENT_PATH = os.path.join(LOG_PATH, EXPERIMENT)
# create evaluator
evaluator = Evaluator(EXPERIMENT_PATH, WORLD_DIR, AGENTS_DICT)
# plot results, paths, qs
evaluator.plot_paths(PLOT_EVERY_KTH_EPOCH, NUM_PLOT_PATHS_IN_ROW)
evaluator.plot_results(AGENTS_DICT.keys(), SLIDING_MEAN_WINDOW)
evaluator.plot_qs(AGENTS_DICT.keys(), "q.tsv")

#evaluator.plot_epsilon()
#evaluator.plot_expansive_spl_radius()
