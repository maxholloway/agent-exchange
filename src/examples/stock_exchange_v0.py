"""
This is an example of an extremely simple exchange, meant
to represent a primitive market-order only system. This
exchange most closely resembles the following scenario:
+ k agents are waiting outside of a room, and none of
  them can communicate with one another
+ each agent has some cash and some of asset A
+ each person writes in pen how many shares of equity
  A that they want (without communicating). Agents can
  put a negative number if they want to sell shares.
+ each agent enters the room, and they all settle
  with each other. That is, every trade that can happen
  among the agents will happen.
+ the remaining desired amount, n, must still get
  purchased. So the agents call on bankers from
  outside of the room to buy these extra shares for
  a premium; alternatively, if the agents wanted to
  sell too much, they call the bankers outside of the
  room to buy their unwanted shares at a discount.
  This premium/discount shows up in the base "price"
  of the asset that the people in the room can charge
  or pay for transfer of the asset.
"""

from typing import Sequence
from src.exchange import Exchange
from src.agent import Agent

import numpy as np
from random import randint

class StockExchangeV0State:
    def __init__(self, price):
        self.price = price
    
    def __repr__(self):
        return f"[{self.price}]"


class StockExchangeV0(Exchange):
    """A simplified stock exchange.
    Here's the specification:
        + only 1 asset is traded
        + the action space is the number of shares the agents will purchase
          of the asset
        + the price of the stock is initially `base` dollars, then it
          increases by epsilon*(num_bought-num_sold); this is an estimate
          of a premium paid for buying in the middle of a buying frenzy,
          and the discount for buying when asset demand is low.
        + the `exchange_state` is just a single float value, denoting the asset price
    """
    def __init__(self, agents: Sequence[Agent], base: float=100.00, epsilon: float=0.01):
        super().__init__(agents)
        self.epsilon = epsilon # sensitivity of price to changes in net-demand
        self.exchange_state = StockExchangeV0State(base)

    def get_exchange_state(self) -> StockExchangeV0State:
        return self.exchange_state

    def update_exchange_state(self, actions: np.array):
        current_price = self.get_exchange_state().price
        net_shares_purchased = np.sum(actions)
        price_change = round(self.epsilon * net_shares_purchased, 2)
        new_price = current_price + price_change
        self.exchange_state.price = new_price # modify the exchange state
        return

    def get_agent_value(self):
        current_price = self.get_exchange_state().price
        return np.array(
            [
                current_price * agent.num_shares + agent.capital
                for agent in self.agents
            ]
        )


class StockAgentV0(Agent):
    def __init__(self, initial_num_shares=1000, initial_capital=100000):
        super().__init__()
        self.num_shares = initial_num_shares
        self.capital = initial_capital
        self.shares_bought_last = 0

    def __str__(self):
        return f"Capital: ${self.capital}\tEquity: {self.num_shares}"

    def __repr__(self):
        return str(self)


class StockAgentV0Random(StockAgentV0):
    def __init__(self, initial_num_shares=1000, initial_capital=100000):
        super().__init__()

    def get_action(self, exchange_state: StockExchangeV0State):
        """Buy or sell a random number of shares within our bounds.
        Keep in mind that a negative integer here is means we go
        short.
        """
        max_num_shares = self.capital // exchange_state.price
        min_num_shares = -self.num_shares
        shares_to_buy = randint(min_num_shares, max_num_shares) # this can be negative
        self.shares_bought_last = shares_to_buy
        return shares_to_buy

    def action_results_update(self, new_exchange_state, reward, done, info):
        """This is where we find out how much we actually paid for the stock.
        Once we find this out, we can adjust how much capital we 
        """
        asset_price = new_exchange_state.price # the price that was actually paid for the asset
        self.capital -= self.shares_bought_last * asset_price
        self.num_shares += self.shares_bought_last


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    agents = np.array([StockAgentV0Random() for _ in range(2)])
    exchange = StockExchangeV0(agents, epsilon=0.0001)
    plotvals = []
    for i in range(10000):
        exchange.simulate_step()
        if i % 300 == 0:
            avg_value = np.mean(exchange.get_agent_value())
            print(f"The average agent has ${avg_value}")
            plotvals.append(avg_value)
    plt.plot(plotvals)
    plt.grid()
    plt.show()






