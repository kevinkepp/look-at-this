from sft.runner.Tester import Tester

exp = "tmp/logs/20170119-013032_exp"
testset_path = "tmp/tester/testworlds"
t = Tester()
t.run_exp(exp, testset_path)
