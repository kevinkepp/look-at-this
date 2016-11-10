from lat.Simulator import SimpleMatrixSimulator as Simulator
from lat.KeyboardAgent import KeyboardAgent
from lat.SimpleReward import GaussianDistanceReward

agent = KeyboardAgent()
agent.set_training_mode(True)
reward = GaussianDistanceReward()
sim = Simulator(agent,reward,5,4,max_steps=30,bounded=False)

sim.run(visible=True,trainingmode=True)
sim.visualize_path()