from sft.runner.Tester import Tester

exp = "/home/philibb/Dropbox/Uni/Robotics Project/github-repo/tmp/logs/20170119-013032_exp"
testset_path = "/home/philibb/Dropbox/Uni/Robotics Project/github-repo/tmp/tester/testworlds"
t = Tester()
t.run_on_exp(exp, testset_path)