from sft.eval.Evaluator import Evaluator

EXPERIMENT_PATH = "/home/philibb/Dropbox/Uni/Robotics Project/github-repo/tmp/logs/20161220-135155_exp1"
WORLD_DIR = "world"
AGENTS_DICT = {
	"Deep[1000,200]": "agent0",
	"Deep[16,8]": "agent1"}

# results
SLIDING_MEAN_WINDOW = 100

# paths
PLOT_EVERY_KTH_EPOCH = 100
NUM_PLOT_PATHS_IN_ROW = 5

# parameters

################################## NOCH NICHT FERTIG ########################################

evaluate = Evaluator(EXPERIMENT_PATH, WORLD_DIR, AGENTS_DICT)

# evaluate.plot_results(AGENTS_DICT.keys(), SLIDING_MEAN_WINDOW)
# evaluate.plot_paths(AGENTS_DICT.keys()[:], PLOT_EVERY_KTH_EPOCH, NUM_PLOT_PATHS_IN_ROW)
