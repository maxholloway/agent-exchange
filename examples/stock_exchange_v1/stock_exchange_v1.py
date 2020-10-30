from random import shuffle, random, randint
from typing import List, Sequence, Tuple, Dict, Union

import numpy as np
import pandas as pd

from agent_exchange.agent import Agent
from agent_exchange.exchange import Exchange

from stock_exchange_v1_utils import dict_repr, dict_str, round_price

EXCHANGE_PRECISION = 2 # 2 digits of precision for stock prices

class StockExchangeV1OrderTypes:
    MARKET="MARKET"
    LIMIT="LIMIT"


class StockExchangeV1Action:
    """Action taken by an agent.
    """
    def __init__(self, order_type, order_size, limit_price=None):
        if order_type == StockExchangeV1OrderTypes.LIMIT:
            assert limit_price != None, "Must specify a limit price for a limit order."
        
        self.order_type = order_type
        self.order_size = order_size
        self.limit_price = limit_price

    def __repr__(self):
        return dict_repr(self)

    def __str__(self):
        return dict_str(self)


class StockExchangeV1FillTicket:
    """This object encapsulates all of the information
    regarding what orders are filled at what prices.
    A fill ticket corresponds to a single agent on a
    single time step for a single asset.
    """
    def __init__(self):
        # Old bids/asks that were filled
        self.closed_bids: Dict[float, int] = {}
        self.closed_asks: Dict[float, int] = {}

        # New bids/asks that were opened
        self.open_bids: Dict[float, int] = {}
        self.open_asks: Dict[float, int] = {}

    def __price_to_quantity_helper(self, member, price, quantity):
        """Increments quantity for a price for a data
        member.
        """
        if price not in member:
            member[price] = 0
        
        member[price] += quantity

    def add_closed_bid(self, price, quantity):
        """Log a closed limit buy order.
        """
        self.__price_to_quantity_helper(self.closed_bids, price, quantity)

    def add_closed_ask(self, price, quantity):
        """Log a closed limit sell order.
        """
        self.__price_to_quantity_helper(self.closed_asks, price, quantity)

    def add_open_bid(self, price, quantity):
        """Log an open limit buy order.
        """
        self.__price_to_quantity_helper(self.open_bids, price, quantity)

    def add_open_ask(self, price, quantity):
        """Log an open limit sell order.
        """
        self.__price_to_quantity_helper(self.open_asks, price, quantity)

    def __repr__(self):
        return dict_repr(self)

    def __str__(self):
        return dict_str(self)


class StockExchangeV1OrderBookSides:
    BIDS = "BIDS"
    ASKS = "ASKS"


class StockExchangeV1OrderBook:
    """Order book implementation.
    The data is stored in the following
    way: two pd.DataFrames, where rows are the price, 
    columns are the agent_id, and values are the 
    number of shares. One of the dataframes will be
    bids, the other dataframe will be asks.
    """
    def __init__(self, n_agents, default_bid=90, default_ask=100):
        self.asks = pd.DataFrame(columns=range(n_agents)) # must be in ascending order
        self.bids = pd.DataFrame(columns=range(n_agents)) # must be in ascending order

        self.default_bid = default_bid
        self.default_ask = default_ask

    def get_bid(self):
        if len(self.bids.index) == 0:
            return self.default_bid
        return round_price(max(self.bids.index), EXCHANGE_PRECISION)

    def get_ask(self):
        if len(self.asks.index) == 0:
            return self.default_ask
        return round_price(min(self.asks.index), EXCHANGE_PRECISION)

    def get_spread(self):
        return round_price(self.get_ask() - self.get_bid(), EXCHANGE_PRECISION)

    def __get_book_from_side(self, book_side: str):
        """Add error checking around accessing the
        order book.

        TODO: understand why this is necessary, and possibly
        remove it.
        """
        if book_side == StockExchangeV1OrderBookSides.ASKS:
            return self.asks
        elif book_side == StockExchangeV1OrderBookSides.BIDS:
            return self.bids
        else:
            raise(Exception("Unknown `book_side`."))

    def __set_book_from_side(self, book_side: str, book: pd.DataFrame):
        if book_side == StockExchangeV1OrderBookSides.ASKS:
            self.asks = book
        elif book_side == StockExchangeV1OrderBookSides.BIDS:
            self.bids = book
        else:
            raise(Exception("Unknown `book_side`."))

    def __take_liquidity(
        self,
        book_side: str, 
        taker_index: int,  
        price: float, 
        num_shares: int,
        order_fills: Dict[int, StockExchangeV1FillTicket]):
        """Fill up to `num_shares` number of shares by
        taking liquidity at a certain `price` on the side
        `book_side`. We then return how many shares were filled.

        We will also delete a price's entry from the book if we 
        take all of the liquidity at that price.

        TODO: test this method
        TODO: clean up the code here... fewer if statements would be nice
        """

        book = self.__get_book_from_side(book_side)
        
        # Shuffle the makers so that we don't have a biased ordering of fills.
        liquidity = book.loc[price].values.copy() # a mapping from market maker index to their amount of liquidity
        maker_index_and_liquidity_amount = list(zip(range(len(liquidity)), liquidity))
        shuffle(maker_index_and_liquidity_amount)
        
        shares_left_to_fill = num_shares
        for maker_index, liquidity_amount in maker_index_and_liquidity_amount:
            assert liquidity_amount == book.loc[price][maker_index], "These should've been the same!"

            # Either take all of the maker's liquidity, OR
            # fill our desired number of shares short of the
            # maker's liquidity. Note: these scenarios
            # aren't mutually exclusive
            if shares_left_to_fill < liquidity_amount:
                shares_exchanged = shares_left_to_fill

                # Update order book and shares left to fill
                book.loc[price][maker_index] -= shares_exchanged
                shares_left_to_fill -= shares_exchanged
            else:
                shares_exchanged = liquidity_amount

                # Update order book and shares left to fill
                book.loc[price][maker_index] -= shares_exchanged
                shares_left_to_fill -= shares_exchanged

            # Update maker and taker fill tickets
            if maker_index not in order_fills:
                order_fills[maker_index] = StockExchangeV1FillTicket()
                
            if taker_index not in order_fills:
                order_fills[taker_index] = StockExchangeV1FillTicket()
                
            if book_side == StockExchangeV1OrderBookSides.ASKS:
                order_fills[maker_index].add_closed_ask(price, shares_exchanged)
                order_fills[taker_index].add_closed_bid(price, shares_exchanged)
            else:
                order_fills[maker_index].add_closed_bid(price, shares_exchanged)
                order_fills[taker_index].add_closed_ask(price, shares_exchanged)
         

        # Clean up book           
        liquidity_left = np.sum(book.loc[price].values)
        if liquidity_left == 0:
            book.drop(price, inplace=True)

        return (num_shares - shares_left_to_fill)

    def __make_liquidity(
        self, 
        book_side: str, 
        maker_index: int, 
        price: float, 
        num_shares: int,
        order_fills: Dict[int, StockExchangeV1FillTicket]):
        """Provide liquidity by deepening the order book with a limit order.
        This is equivalent to placing a limit order that is non-market-clearing.

        If `num_shares` is 0, then we don't modify the book.

        TODO: unit tests
        """

        if num_shares == 0:
            return

        # Update the book
        book = self.__get_book_from_side(book_side)
        if price not in book.index:
            # insert price into book.index, maintaining sorted order
            new_index = book.index.insert(book.index.searchsorted(price), price)
            new_book = book.reindex(new_index, fill_value=0)
            # print(f"New book:\n{new_book}")
            self.__set_book_from_side(book_side, new_book)
            book = self.__get_book_from_side(book_side)
        
        book[maker_index][price] += num_shares
        # print(f"Book after adding entry:\n{book}")
        # print(f"Book showing up in actual state:\n{self.__get_book_from_side(book_side)}")

        # Update order_fills
        if maker_index not in order_fills:
            order_fills[maker_index] = StockExchangeV1FillTicket()

        if book_side == StockExchangeV1OrderBookSides.ASKS:
            order_fills[maker_index].add_open_ask(price, num_shares)
        else:
            order_fills[maker_index].add_open_bid(price, num_shares)

    def update_with_order(
        self, 
        agent_index: int, 
        order: StockExchangeV1Action, 
        order_fills: Dict[int, StockExchangeV1FillTicket]) -> Tuple:
        """Return a tuple of how many shares we filled,
        the total fill cost or fill revenue, and a dictionary
        mapping from agent_index to a fill ticket.

        agent_index: the index of the agent who places the order

        TODO: unit tests
        """
        shares_filled = 0

        if order.order_size > 0:
            # Fill all of the shares that we can
            ask_prices = self.asks.index

            order_volume = order.order_size

            if order.order_type == StockExchangeV1OrderTypes.MARKET:
                limit_price = float('inf')
            elif order.order_type == StockExchangeV1OrderTypes.LIMIT:
                limit_price = order.limit_price
            else:
                raise(Exception(f'Invalid order type "{order.order_type}"'))

            # print(f"Limit price: {limit_price}")
            ask_fill_max_index = ask_prices.searchsorted(limit_price, "right") # index where we can no longer keep filling
            for fill_index in range(ask_fill_max_index):
                fill_price = ask_prices[fill_index]
                unfilled_shares = order_volume - shares_filled
                shares_filled += self.__take_liquidity(
                    StockExchangeV1OrderBookSides.ASKS, # we're market-buying, so we take liquidity from the ask book side
                    agent_index, 
                    fill_price, 
                    unfilled_shares,
                    order_fills)

            # Place the unfilled shares on the bids book (only if it's a limit order)
            if order.order_type == StockExchangeV1OrderTypes.LIMIT:
                unfilled_shares = order_volume - shares_filled
                self.__make_liquidity(
                    StockExchangeV1OrderBookSides.BIDS, 
                    agent_index, 
                    order.limit_price, 
                    unfilled_shares,
                    order_fills)
        elif order.order_size < 0:
            # Fill all of the shares that we can on the bids book
            bid_prices = self.bids.index

            order_volume = -order.order_size

            if order.order_type == StockExchangeV1OrderTypes.MARKET:
                limit_price = 0 # sell as low as the book goes, but if the book is depleted, stop selling
            elif order.order_type == StockExchangeV1OrderTypes.LIMIT:
                limit_price = order.limit_price
            else:
                raise(Exception(f'Invalid order type "{order.order_type}"'))
            
            bid_fill_min_index = bid_prices.searchsorted(limit_price, "left")
            for fill_index in range(len(bid_prices)-1, bid_fill_min_index-1, -1):
                fill_price = bid_prices[fill_index]
                unfilled_shares = order_volume-shares_filled
                shares_filled += self.__take_liquidity(
                    StockExchangeV1OrderBookSides.BIDS, 
                    agent_index, 
                    fill_price, 
                    unfilled_shares,
                    order_fills)

            # Place the unfilled shares on the asks book (only if it's a limit order)
            if order.order_type == StockExchangeV1OrderTypes.LIMIT:
                unfilled_shares = order_volume - shares_filled
                self.__make_liquidity(
                    StockExchangeV1OrderBookSides.ASKS,
                    agent_index,
                    order.limit_price,
                    unfilled_shares,
                    order_fills)

        # print(f"Ask book:\n{self.asks}.\nTaking liquidity with {order}.\n")
        # print(f"Bid book:\n{self.bids}.\nTaking liquidity with {order}.\n")
        # print(f"Filled {shares_filled} shares when taking liquidity")

        return

    def __repr__(self):
        return dict_repr(self)

    def __str__(self):
        return dict_str(self)


class StockExchangeV1(Exchange):
    """Single-stock exchange that uses an 
    order book as its store of state.
    """

    def __init__(self, agents: Sequence[Agent]):
        super().__init__(agents)
        self.order_book: StockExchangeV1OrderBook = StockExchangeV1OrderBook(len(agents), 99.95, 100.05)
        self.order_fills_this_step: Dict = {}
        
        # Chronological list of max-bid and min-ask at the end of each step -- can be updated in the `on_step_end` method
        self.bids = []
        self.asks = []

    def get_order_book(self) -> StockExchangeV1OrderBook:
        return self.order_book

    def get_bids(self) -> List[float]:
        """Return the a list of end-of-step
        maximum bids, ordered chronologically.

        Returns:
            List[float]: a list of end-of-step maximum bids, ordered chronologically.
        """
        return self.bids

    def get_asks(self) -> List[float]:
        """Return the a list of end-of-step minimum asks,
        ordered chronologically.

        Returns:
            List[float]: a list of end-of-step minimum asks, 
            ordered chronologically.
        """
        return self.asks

    def get_exchange_state(self) -> StockExchangeV1OrderBook:
        """We define the state of the exchange
        as the state of the order book.
        """
        return self.get_order_book()
    def get_info(self, agent_index):
        """Relay information about how many shares 
        of the user's previous orders filled.
        """
        if agent_index in self.order_fills_this_step:
            return self.order_fills_this_step[agent_index]
        else:
            return None
    
    def on_step_end(self):
        """Clean up the state from
        order_fills_this_step
        """
        self.order_fills_this_step = {}
        self.bids.append(self.order_book.get_bid())
        self.asks.append(self.order_book.get_ask())
        return

    def update_exchange_state(self, orders: np.array):
        """We will go through the orders in a random order,
        filling them in that order. If multiple limit orders 
        stack on the same price, we randomly choose which 
        order is executed first.
        """

        agent_and_order = list(zip(range(len(orders)), orders))
        shuffle(agent_and_order) # process agent actions in random sequence
        for agent_index, order in agent_and_order:
            # Try to fill the agent's order, and update
            # the order book in the process.
            # print(f"`update_exchange_state`, processing agent {agent_index}")
            self.order_book.update_with_order(agent_index, order, self.order_fills_this_step)
        
        return