import os

from sft.runner.Tester import Tester

log_dir = "tmp/logs/"
exp = log_dir + sorted(os.listdir(log_dir))[-1]
testset_path = "tmp/tester/testworlds"
t = Tester()
t.run_on_exp(exp, testset_path)
t.plot_results(exp)

# used for getting the q values (estimated by a certain model of an agent) for one state
if False:
	state_path = "tmp/tester/teststate/test_state"
	world_path = "tmp/logs/20170119-145057_exp/world/world.py"
	agent_path = "tmp/logs/20170119-145057_exp/agent_ah10_256/agent.py"
	model_path = "tmp/logs/20170119-145057_exp/agent_ah10_256/models/model_2900.h5.npz"
	b = Tester()
	s = b.get_q_one_state(testset_path, world_path, agent_path, model_path)