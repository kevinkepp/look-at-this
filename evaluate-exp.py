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

evaluator = Evaluator(EXPERIMENT_PATH, WORLD_DIR, AGENTS_DICT)

evaluator.plot_paths(PLOT_EVERY_KTH_EPOCH, NUM_PLOT_PATHS_IN_ROW)

evaluator.plot_results(AGENTS_DICT.keys(), SLIDING_MEAN_WINDOW)

evaluator.load_epsilon()
evaluator.load_expansive_spl_radius()
evaluator.plot_parameters()

evaluator.load_qs()
evaluator.plot_qs()