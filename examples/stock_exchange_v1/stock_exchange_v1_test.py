import inspect
from typing import Tuple
from .stock_exchange_v1 import StockExchangeV1OrderBook, StockExchangeV1Action, StockExchangeV1OrderTypes, StockAgentV1InternalState, StockExchangeV1FillTicket

class OrderBookTests:

    def buildTest():
        ob = StockExchangeV1OrderBook(1)

    def testBasicFunctionality():
        """Tests basic functionality for placing bids and asks.
        """
        ob = StockExchangeV1OrderBook(2)

        buyer_index = 0
        for higher_bid in range(1, 10):
            ob.update_with_order(
                    buyer_index,
                    StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, 10, higher_bid), # limit buy 10 shares at $1
                    {} # order fills; only necessary for larger-scale state tracking
                )
            assert ob.get_bid() == higher_bid


        seller_index = 1
        for lower_ask in range(20, 10, -1):
            ob.update_with_order(
                    seller_index,
                    StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, -10, lower_ask),
                    {}
                )
            assert ob.get_ask() == lower_ask

    def testShallowAskBookOnMarketOrder():
        ob = StockExchangeV1OrderBook(2)
        seller_index, buyer_index = 0, 1

        for ask in range(1, 11):
            ob.update_with_order(
                    seller_index,
                    StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, -1, ask),
                    {}
                )

        ## Buy 6 shares at market price
        ob.update_with_order(
                buyer_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, 6),
                {}
            )

        assert ob.get_ask() == 7

    def testShallowBidBookOnMarketOrder():
        ob = StockExchangeV1OrderBook(2)
        buyer_index, seller_index = 0, 1    

        for ask in range(1, 11):
            ob.update_with_order(
                    buyer_index,
                    StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, 1, ask),
                    {}
                )

        ## Sell 6 shares at the market
        ob.update_with_order(
                seller_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, -6),
                {}
            )

        assert ob.get_bid() == 4 # since prices [10, 9, 8, 7, 6, 5] were executed
    
    def testDeepBidBookOnMarketOrder():
        ob = StockExchangeV1OrderBook(2)
        buyer_index, seller_index = 0, 1  

        small_bid, small_bid_position = 0.5, 1
        ob.update_with_order(
                buyer_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, small_bid_position, small_bid),
                {}
            )

        large_bid, large_bid_position = 1, 1000
        ob.update_with_order(
                buyer_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, large_bid_position, large_bid),
                {}
            )


        market_sell_size1 = -350
        ob.update_with_order(
                seller_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, market_sell_size1),
                {}
            )

        assert ob.get_bid() == large_bid # bid is unaffected by this market order

        market_sell_size2 = -649
        ob.update_with_order(
                seller_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, market_sell_size2),
                {}
            )

        assert ob.get_bid() == large_bid # bid is unaffected by this market order

        market_sell_size3 = -1
        ob.update_with_order(
                seller_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, market_sell_size3),
                {}
            )


        assert ob.get_bid() == small_bid # exhaust the very last share of the large bid

    def testDeepAskBookOnMarketOrder():
        ob = StockExchangeV1OrderBook(2)
        seller_index, buyer_index= 0, 1  

        large_ask, large_ask_position = 1, -1
        ob.update_with_order(
                buyer_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, large_ask_position, large_ask),
                {}
            )

        small_ask, small_ask_position = .5, -1000
        ob.update_with_order(
                buyer_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, small_ask_position, small_ask),
                {}
            )


        market_buy_size1 = 350
        ob.update_with_order(
                seller_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, market_buy_size1),
                {}
            )

        assert ob.get_ask() == small_ask # ask is unaffected by this market order

        market_buy_size2 = 649
        ob.update_with_order(
                seller_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, market_buy_size2),
                {}
            )

        assert ob.get_ask() == small_ask # ask is unaffected by this market order

        market_buy_size3 = 1
        ob.update_with_order(
                seller_index,
                StockExchangeV1Action(StockExchangeV1OrderTypes.MARKET, market_buy_size3),
                {}
            )


        assert ob.get_ask() == large_ask # exhaust the very last share of the large bid

    def testMultiFillOrder():
        n_agents = 4
        ob = StockExchangeV1OrderBook(n_agents)
        agent1, agent2, agent3, agent4 = range(n_agents)

        # Agents 1-3 place limit buy orders
        for agent in {agent1, agent2, agent3}:
            ob.update_with_order(
                    agent,
                    StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, 10, 1),
                    {}
                )

        # Agent 4 places market sell order
        ob.update_with_order(
                agent1,
                StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, -6, 1),
                {}
            )

        assert ob.get_bid() == 1

        ob.update_with_order(
                agent1,
                StockExchangeV1Action(StockExchangeV1OrderTypes.LIMIT, -23, 1),
                {}
            )

        assert ob.get_bid() == 1


class InternalStateTests:
    def testUpdateInternalStateBasic():
        INITIAL_POS, INITIAL_CAP = 100, 10000
        state = StockAgentV1InternalState(INITIAL_POS, INITIAL_CAP)
        assert state.get_num_shares() == INITIAL_POS
        assert state.get_capital() == INITIAL_CAP

        # Place a limit sale
        sale_price, sale_position = 30.12, 10
        ft = generate_pseudo_fill_ticket(open_ask=(sale_price, sale_position)) # ticket indicating the limit sale of 100 shares as $30.12
        state.update_with_fill_ticket(ft)

        # Capital and position should not change if we place a limit order that's not filled yet
        assert state.get_capital() == INITIAL_CAP
        assert state.get_num_shares() == INITIAL_POS

        # Limit sale gets partially filled
        shares_filled1 = 3
        ft = generate_pseudo_fill_ticket(closed_ask=(sale_price, shares_filled1))
        state.update_with_fill_ticket(ft)

        # Position should decrease by shares_filled1 shares, capital should increase accordingly -- this may become outdated if we add transaction cost to the mix
        assert state.get_num_shares()   == INITIAL_POS - shares_filled1
        assert state.get_capital()      == INITIAL_CAP + shares_filled1*sale_price

        return




    pass

def generate_pseudo_fill_ticket(
        closed_bid: Tuple[float, int]=None, 
        closed_ask: Tuple[float, int]=None, 
        open_bid: Tuple[float, int]=None, 
        open_ask: Tuple[float, int]=None):
    """
    Args:
        closed_bid (Tuple[float, int], optional): bid price and number of shares filled. Defaults to None.
        closed_ask (Tuple[float, int], optional): ask price and number of shares filled. Defaults to None.
        open_bid (Tuple[float, int], optional): bid price and number of shares in the position. Defaults to None.
        open_ask (Tuple[float, int], optional): ask price and number of shares in the position. Defaults to None.
    """
    ft = StockExchangeV1FillTicket()
    if closed_bid:
        ft.add_closed_bid(*closed_bid)
    if closed_ask:
        ft.add_closed_ask(*closed_ask)
    if open_bid:
        ft.add_open_bid(*open_bid)
    if open_ask:
        ft.add_open_ask(*open_ask)
    print(f'fill ticket: {ft}')
    return ft

def test():
    failed_tests = []
    n_tests = 0
    for cls in [OrderBookTests, InternalStateTests]:
        for funcName, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            # print(funcName)
            lowercase = funcName.lower()
            if 'test' in lowercase:
                try:
                    func()
                except Exception as ex:
                    print(ex)
                    failed_tests.append(test)
            n_tests += 1

    n_failed_tests = len(failed_tests)
    n_successful_tests = n_tests - n_failed_tests
    print(f"""
        Num successful tests: {n_successful_tests}
        Num failed tests: {n_failed_tests}
        Failed tests: {failed_tests}
    """)
    

if __name__ == '__main__':
    test()
















