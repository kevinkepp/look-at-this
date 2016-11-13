from lat.Simulator import SimpleMatrixSimulator as Simulator
from lat.KeyboardAgent import KeyboardAgent
from lat.SimpleReward import LinearReward as Reward

agent = KeyboardAgent()
reward = Reward()
sim = Simulator(agent,reward,5,4,max_steps=30,bounded=False)

sim.run(visible=True,trainingmode=True)
# sim.visualize_path()