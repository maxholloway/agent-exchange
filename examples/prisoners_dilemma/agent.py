from typing import Dict, Tuple

from agent_exchange.agent import Agent
from utils import BufferList


class Actions:
    DEFECT = 0
    NO_DEFECT = 1


class PrisonersDilemmaBaseAgent(Agent):
    def __init__(self, memorylen=1):
        super().__init__()
        self.historical_rewards = BufferList(memorylen)
        self.historical_actions = BufferList(memorylen)

    def action_results_update(self, new_exchange_state, reward, done, info):
        self.historical_rewards.append(reward)

    def get_action(self, exchange_state):
        raise(NotImplementedError("Be sure to update `self.historical_actions` with whatever action is taken."))

