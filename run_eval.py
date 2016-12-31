from sft.eval.visual.PathVisualizer import PathVisualizer

# visualize paths based on output directory
print("RuntimeWarnings related to imports can be ignored")
PathVisualizer().visualize_paths("tmp/logs/20161229-174200_exp", "sft.config.exp", "no-cloning")
