Stock exchange simulator for a single stock.
We define the stock exchange environment here
with the following specification:
    1. The state of the exchange is
    given by an entire order book.
    2. Agents are placed arbitrarily
    into the order book. When an agent
    places bid at the same price as
    another agent, the order of fill is
    undefined behavior.
    3. Agents' action space just comprises
    of the expressivity of an order object.
    An order object can be a market or a
    limit order.

This exchange will still have the following
limitations:
    1. All agents get to view the order
       book at the same time, and all
       agents have arbitrary priority
       in the order book; this neglects
       the existence of HFTs       
    2. There's only a single stock traded.
    3. Scalability. The order book could
       become quite large in real life, 
       and we won't be able to do scale well
       with a huge order book here.
    4. No spreads or transaction costs.
    5. Naive cold starting. We cold start with
       an empty order book, which may or may
       not affect agents' policies. When getting
       a bid or an ask quote from the empty order
       book, we return a fall-back default bid
       or ask.
