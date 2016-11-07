from lat.Simulator import SimpleMatrixSimulator as Simulator
from lat.KeyboardAgent import KeyboardAgent
from lat.SimpleReward import LinearReward

agent = KeyboardAgent()
agent.set_training_mode(True)
reward = LinearReward()
sim = Simulator(agent,reward,3,4,max_steps=30,bounded=False)

sim.run(visible=True,trainingmode=True)