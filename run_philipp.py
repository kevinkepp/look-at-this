from reward.SimpleReward import LinearReward as Reward
from sft.agent.KeyboardAgent import KeyboardAgent
from sft.eval.SimpleVisualize import PlotMatrix as Visual
from sft.sim.Simulator import GaussSimulator as Simulator

agent = KeyboardAgent()
reward = Reward()
visual = Visual()
sim = Simulator(agent,reward,3,3,max_steps=30,visualizer=visual,bounded=False)

sim.run(visible=True,trainingmode=True,epochs=3)
# sim.visualize_path()