from Simulator import SimpleMatrixSimulator as Simulator
from KeyboardAgent import KeyboardAgent
from SimpleReward import LinearReward

agent = KeyboardAgent()
agent.set_training_mode(True)
reward = LinearReward()
sim = Simulator(agent,reward,3,1,max_steps=20,bounded=False)

sim.run(visible=True)
# import readchar
# print("hit some key")
# a = readchar.readkey()
# print(a)
# print("end of program")