import numpy as np
from typing import Callable, Sequence
from agent_exchange.agent import Agent

from agent_exchange.exchange import Exchange
from utils import BufferList

class PrisonersDilemmaExchange(Exchange):
    """
    Defines the exchange for a simple prisoner's
    dilemma game. In each round, an agent will take
    an action, and the exchange will distribute rewards.
    In this case, there are no state transitions.
    """

    def __init__(self, agents: Sequence[Agent], reward_fn: Callable[[Sequence], Sequence]) -> None:
        super().__init__(agents)
        self.__player_actions = []
        self.__player_rewards = []
        self.__reward_fn = reward_fn

    def update_exchange_state(self, actions: np.array):
        self.__player_actions = actions
        self.__player_rewards = self.__reward_fn(actions)

    def get_exchange_state(self):
        return None

    def get_reward(self, agent_index):
        return self.__player_rewards[agent_index]

    def get_info(self, agent_index):
        adversary_actions = []
        for i, action in enumerate(self.__player_actions):
            if i == agent_index:
                continue
            adversary_actions.append(action)
        return adversary_actions

    def on_step_end(self):
        print(f"The rewards were {self.__player_rewards} in round {self.t-1}.")


if __name__ == "__main__":

    def simple_reward_fn(twoPlayersActions):
        """Make a simple prisoners dilemma
        where the payoff matrix is as follows

        1 \ 2
                defect      no defect
        defect    (-7, -7)    (0, -10)
        no defect (-10, 0)    (-3, -3)

        """
        actionA, actionB = twoPlayersActions
        if actionA == actionB:
            if actionA == 0:
                return np.array([-7, -7])
            else:
                return np.array([-3, -3])
        elif actionA == 1:
            return np.array([-10, 0])
        else:
            return np.array([0, -10])

    class Altruist(Agent):
        def get_action(self, exchange_state):
            return 1

    couple_of_altruists = [Altruist() for _ in range(2)]
    SimplePrisonersDilemmaExchange = PrisonersDilemmaExchange(
        couple_of_altruists, simple_reward_fn
    )

    SimplePrisonersDilemmaExchange.simulate_steps(10)
