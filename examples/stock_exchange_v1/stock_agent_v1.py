"""
Provide basic functionality for stock agent to interact with exchange.
"""
from typing import Union

from agent_exchange.agent import Agent

from stock_exchange_v1_utils import dict_repr, dict_str
from stock_exchange_v1 import StockExchangeV1FillTicket, StockExchangeV1OrderBook


class StockAgentV1InternalState:
    def __init__(self, initial_num_shares: int, initial_capital: float):
        self.num_shares = [initial_num_shares]
        self.capital = [initial_capital]
        
        # Mappings from price to number of shares
        self.open_bids = {}
        self.open_asks = {}

    def on_timestep_passed(self, fill_ticket: Union[type(None), StockExchangeV1FillTicket]):
        if fill_ticket != None:
            new_num_shares, new_capital = self.update_with_fill_ticket(fill_ticket)
        else:
            new_num_shares, new_capital = self.get_num_shares(), self.get_capital()
            
        self.num_shares.append(new_num_shares)
        self.capital.append(new_capital)
        
        
        
    def update_with_fill_ticket(self, ticket: StockExchangeV1FillTicket):
        """Use the fill ticket to update our state variables.
        """
        
        # The state updates for after the update -- this should be modified in this function when capital or num_shares changes
        new_num_shares = self.get_num_shares()
        new_capital = self.get_capital()

        # Add new bids
        for price in ticket.open_bids:
            StockAgentV1InternalState.increment_or_create(self.open_bids, price, ticket.open_bids[price])

        # Add new asks
        for price in ticket.open_asks:
            StockAgentV1InternalState.increment_or_create(self.open_asks, price, ticket.open_asks[price])

        # Remove old bids that were filled in the past time step
        for price in ticket.closed_bids:
            shares_bought = ticket.closed_bids[price]
            new_num_shares += shares_bought
            new_capital -= price * shares_bought
            StockAgentV1InternalState.decrement_and_try_delete(self.open_bids, price, shares_bought)
            

        # Remove old asks that were filled in the past time step
        for price in ticket.closed_asks:
            shares_sold = ticket.closed_asks[price]
            new_num_shares -= shares_sold
            new_capital += price * shares_sold
            StockAgentV1InternalState.decrement_and_try_delete(self.open_asks, price, shares_sold)
            
        return new_num_shares, new_capital

    def get_num_shares(self):
        return self.num_shares[-1]
    
    def get_capital(self):
        return self.capital[-1]

    def __repr__(self):
        return dict_repr(self)

    def __str__(self):
        return dict_str(self)

    def increment_or_create(D, key, value):
        """If the key-value pair does not exist yet,
        then add a new key-value pair with `value`
        as the value. Otherwise, increment the
        key's value with `value`.
        """
        key = round(key, 2)
        if key not in D:
            D[key] = 0
        D[key] += value
        if D[key] == 0:
            del D[key]

    def decrement_and_try_delete(D, key, value):
        """Decrement a value in a dictionary,
        and if the new value is 0, then delete
        the k-v pair from the dictionary.
        """
        key = round(key, 2)
        if key not in D:
            D[key] = 0
        D[key] -= value

        if D[key] == 0:
            del D[key]


class StockAgentV1(Agent):
    """A base stock trading agent; this agent itself will perform no-ops each iteration.
    """
    def __init__(self, initial_num_shares, initial_capital):
        super().__init__()
        self.internal_state = StockAgentV1InternalState(initial_num_shares, initial_capital)

    def action_results_update(
        self, 
        new_order_book: StockExchangeV1OrderBook, 
        reward, 
        done: bool, 
        fill_ticket: Union[type(None), StockExchangeV1FillTicket]):
        
        self.internal_state.on_timestep_passed(fill_ticket)
        
    def __repr__(self):
        return dict_repr(self)

    def __str__(self):
        return dict_str(self)