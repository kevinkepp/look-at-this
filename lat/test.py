from Simulator import Simulator
from KeyboardAgent import KeyboardAgent

agent = KeyboardAgent()
sim = Simulator(agent,None,5,5,max_steps=3)

sim.run(visible=True)
# import readchar
# print("hit some key")
# a = readchar.readkey()
# print(a)
# print("end of program")