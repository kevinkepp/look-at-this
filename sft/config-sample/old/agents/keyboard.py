import sft.eps.Constant
import sft.agent.KeyboardAgent
import sft.reward.TargetMiddle
from sft.log.AgentLogger import AgentLogger
from sft import world
from ..world import *

logger = AgentLogger(__name__)

epsilon_update = sft.eps.Constant.Constant(0)  # irrelevant for keyboard agent

agent = sft.agent.KeyboardAgent.KeyboardAgent()

reward = sft.reward.TargetMiddle.TargetMiddle(
	reward_target=1,
	reward_not_target=0
)
