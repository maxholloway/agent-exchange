from .stock_exchange_v1 import StockExchangeV1OrderBook, StockExchangeV1Action, StockExchangeV1OrderTypes

class OrderBookTests:

    def buildTest():
        ob = StockExchangeV1OrderBook(1)

    def testDefaults():
        ob = StockExchangeV1OrderBook(1)

        assert ob.get_bid() == 90
        assert ob.get_ask() == 100

        return 0

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
        ob = StockExchangeV1OrderBook(3)
        agent1, agent2, agent3, agent4 = range(4)

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
        

def test():
    failed_tests = []
    tests = []
    for funcName, func in inspect.getmembers(OrderBookTests, predicate=inspect.isfunction):
        # print(funcName)
        lowercase = funcName.lower()
        if 'test' in lowercase and 'meta':
            tests.append(func)

    for test in tests:
        try:
            test()
        except Exception as ex:
            print(ex)
            failed_tests.append(test)

    n_failed_tests = len(failed_tests)
    n_successful_tests = len(tests) - n_failed_tests
    print(f"""\n
        Successful tests: {n_successful_tests}
        Failed tests: {n_failed_tests}
        {failed_tests}
    """)
    

if __name__ == '__main__':
    import inspect
    test()
















