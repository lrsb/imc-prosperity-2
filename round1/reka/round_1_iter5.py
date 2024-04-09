from datamodel import Trade, Listing, OrderDepth, UserId, TradingState, Order
from typing import List

def StarTrader(state: TradingState) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 2:
            return 10
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 2:
            return -10
        else:
            return -20
    
    symbol = "STARFRUIT"
 
    # Extract the relevant info from the trading_state
    sold_position = state.position.get(symbol,0)
    bought_position = state.position.get(symbol,0)
    bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
    ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))

    # Use the outermost prices as ref_price
    ref_price = ask_book[-1][0] -3.5 if abs(ask_book[-1][1]) > (bid_book[-1][1]) else  bid_book[-1][0] +3.5 

    # Placeholder for the emitted orders
    result = []

    # Removing stale orders
    n_ask_book = []
    for price, qty in ask_book:
        curr_profit = ref_price - price
        # we never take -EV trade, and the max positive exposure that we want to
        # hold depends on the buy_schedule
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position:
            executed_buy = min(buy_schedule(curr_profit) - bought_position, -qty)
            result.append(Order(symbol, price, executed_buy))
            bought_position += executed_buy
            if executed_buy != -qty:
                n_ask_book.append((price, qty+executed_buy))
        else:
            n_ask_book.append((price, qty))
    ask_book = n_ask_book

    n_bid_book = []
    for price, qty in bid_book:
        curr_profit =  price - ref_price
        # we never take -EV trade, and the max negative exposure that we want to
        # hold depends on the sell_schedule
        if curr_profit >= 0 and sell_schedule(curr_profit) < sold_position:
            executed_sell = min( sold_position - sell_schedule(curr_profit), qty)
            result.append(Order(symbol, price, -executed_sell))
            sold_position -= executed_sell
            if executed_sell != qty:
                n_bid_book.append((price, qty-executed_sell))
        else:
            n_bid_book.append((price, qty))
    bid_book = n_bid_book

    # Placing limit orders
    for price, qty in bid_book:
        curr_profit = ref_price - (price+1)
        # we never take -EV trade, and the max positive exposure that we want to
        # hold depends on the buy_schedule
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position:
            placed_buy = buy_schedule(curr_profit) - bought_position
            result.append(Order(symbol, price+1, placed_buy))
            bought_position += placed_buy

    for price, qty in ask_book:
        curr_profit =  (price-1) - ref_price
        # we never take -EV trade, and the max negative exposure that we want to
        # hold depends on the sell_schedule
        if curr_profit >= 0 and sell_schedule(curr_profit) < sold_position:
            placed_sell = sold_position-sell_schedule(curr_profit)
            result.append(Order(symbol, price-1, -placed_sell))
            sold_position -= placed_sell
    return result

def AmetTrader(state: TradingState) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 1.5:
            return 18
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 1.5:
            return -18
        else:
            return -20
    
    symbol = "AMETHYSTS"
 
    # Extract the relevant info from the trading_state
    sold_position = state.position.get(symbol,0)
    bought_position = state.position.get(symbol,0)
    bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
    ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))
    
    # Use the outermost prices as ref_price
    ref_price = ask_book[-1][0] -5. if abs(ask_book[-1][1]) > (bid_book[-1][1]) else  bid_book[-1][0] +5.

    # Placeholder for the emitted orders
    result = []

    # Removing stale orders
    n_ask_book = []
    for price, qty in ask_book:
        curr_profit = ref_price - price
        # we never take -EV trade, and the max positive exposure that we want to
        # hold depends on the buy_schedule
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position:
            executed_buy = min(buy_schedule(curr_profit) - bought_position, -qty)
            result.append(Order(symbol, price, executed_buy))
            bought_position += executed_buy
            if executed_buy != -qty:
                n_ask_book.append((price, qty+executed_buy))
        else:
            n_ask_book.append((price, qty))
    ask_book = n_ask_book

    n_bid_book = []
    for price, qty in bid_book:
        curr_profit =  price - ref_price
        # we never take -EV trade, and the max negative exposure that we want to
        # hold depends on the sell_schedule
        if curr_profit >= 0 and sell_schedule(curr_profit) < sold_position:
            executed_sell = min( sold_position - sell_schedule(curr_profit), qty)
            result.append(Order(symbol, price, -executed_sell))
            sold_position -= executed_sell
            if executed_sell != qty:
                n_bid_book.append((price, qty-executed_sell))
        else:
            n_bid_book.append((price, qty))
    bid_book = n_bid_book

    # Placing limit orders
    for price, qty in bid_book:
        curr_profit = ref_price - (price+1)
        # we never take -EV trade, and the max positive exposure that we want to
        # hold depends on the buy_schedule
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position:
            placed_buy = buy_schedule(curr_profit) - bought_position
            result.append(Order(symbol, price+1, placed_buy))
            bought_position += placed_buy

    for price, qty in ask_book:
        curr_profit =  (price-1) - ref_price
        # we never take -EV trade, and the max negative exposure that we want to
        # hold depends on the sell_schedule
        if curr_profit >= 0 and sell_schedule(curr_profit) < sold_position:
            placed_sell = sold_position-sell_schedule(curr_profit)
            result.append(Order(symbol, price-1, -placed_sell))
            sold_position -= placed_sell
    return result

# Main class
class Trader:    
    def run(self, state: TradingState):
        result = {}
        
        result["STARFRUIT"]= StarTrader(state)
        result["AMETHYSTS"]= AmetTrader(state)

        # String value holding Trader state data required. 
        traderData = "" 
        # Sample conversion request. Check more details below. 
        conversions = None
        return result, conversions, traderData