from lat.Simulator import GaussSimulator as Simulator
from lat.KeyboardAgent import KeyboardAgent
from lat.SimpleReward import LinearReward as Reward
from lat.SimpleVisualize import PlotMatrix as Visual

agent = KeyboardAgent()
reward = Reward()
visual = Visual()
sim = Simulator(agent,reward,3,3,max_steps=30,visualizer=visual,bounded=False)

sim.run(visible=True,trainingmode=True,epochs=3)
# sim.visualize_path()