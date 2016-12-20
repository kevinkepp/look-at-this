import sft.eps.Constant
import sft.agent.KeyboardAgent
import sft.reward.TargetMiddle
from sft.Logger import Logger
from ..world import *

logger = Logger(__name__)

epsilon_update = sft.eps.Constant.Constant(0)  # irrelevant

agent = sft.agent.KeyboardAgent.KeyboardAgent()

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=100,
	reward_not_target=0
)
