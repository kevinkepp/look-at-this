from lat.Simulator import SimpleMatrixSimulator as Simulator
from lat.KeyboardAgent import KeyboardAgent
from lat.SimpleReward import LinearReward as Reward
from lat.SimpleVisualize import PlotMatrix as Visual

agent = KeyboardAgent()
reward = Reward()
visual = Visual()
sim = Simulator(agent,reward,5,4,max_steps=30,visualizer=visual,bounded=False)

sim.run(visible=True,trainingmode=True)
# sim.visualize_path()