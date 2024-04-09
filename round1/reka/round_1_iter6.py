from datamodel import Trade, Listing, OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle

class StarTraderState:
    def __init__(self):
        self.past_pred =None
        self.past_price = None

def StarTrader(state: TradingState, estimator : StarTraderState) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 2.0:
            return 8
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 2.0:
            return -8
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
    # We are betting that the delta of ref price is MA(1) : y[t] = e[t]-0.3*e[t-1]
    if estimator.past_price is not None:
        realized_delta = ref_price - estimator.past_price
        estimator.past_price = ref_price
        if estimator.past_pred is not None:
            new_pred = -0.3*(realized_delta-estimator.past_pred)
            ref_price += new_pred
            estimator.past_pred = new_pred
        else:
            estimator.past_pred = -0.3*realized_delta
    else:
        estimator.past_price = ref_price 

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
    # in some iterations, we are mising the large outer bid. In this case, mock the large outer bid
    if bid_book[-1][1] < 10:
        bid_book.append((bid_book[-1][0]-1, 25))
    for price, qty in bid_book:
        curr_profit = ref_price - (price+1)
        # we never take -EV trade, and the max positive exposure that we want to
        # hold depends on the buy_schedule
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position:
            placed_buy = buy_schedule(curr_profit) - bought_position
            result.append(Order(symbol, price+1, placed_buy))
            bought_position += placed_buy

    # in some iterations, we are mising the large outer ask. In this case, mock the large outer ask
    if ask_book[-1][1] > -10:
        ask_book.append((ask_book[-1][0]+1, -25))
    for price, qty in ask_book:
        curr_profit =  (price-1) - ref_price
        # we never take -EV trade, and the max negative exposure that we want to
        # hold depends on the sell_schedule
        if curr_profit >= 0 and sell_schedule(curr_profit) < sold_position:
            placed_sell = sold_position-sell_schedule(curr_profit)
            result.append(Order(symbol, price-1, -placed_sell))
            sold_position -= placed_sell
    return result, estimator

def AmetTrader(state: TradingState) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 3.5:
            return 11
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 3.5:
            return -11
        else:
            return -20
    
    symbol = "AMETHYSTS"
 
    # Extract the relevant info from the trading_state
    sold_position = state.position.get(symbol,0)
    bought_position = state.position.get(symbol,0)
    bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
    ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))
    
    # Use the value given by the parrot
    ref_price = 10000

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
    # in some iterations, we are mising the large outer bid. In this case, mock the large outer bid
    if bid_book[-1][1] < 10:
        bid_book.append((bid_book[-1][0]-1, 25))
    for price, qty in bid_book:
        curr_profit = ref_price - (price+1)
        # we never take -EV trade, and the max positive exposure that we want to
        # hold depends on the buy_schedule
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position:
            placed_buy = buy_schedule(curr_profit) - bought_position
            result.append(Order(symbol, price+1, placed_buy))
            bought_position += placed_buy

    # in some iterations, we are mising the large outer ask. In this case, mock the large outer ask
    if ask_book[-1][1] > -10:
        ask_book.append((ask_book[-1][0]+1, -25))
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
        
        # Currently only starfruit uses the trading data
        if state.traderData == "":
            star_state = StarTraderState()
        else:
            star_state = jsonpickle.decode(state.traderData)
        
        result["STARFRUIT"], star_state= StarTrader(state, star_state)
        result["AMETHYSTS"]= AmetTrader(state)

        # String value holding Trader state data required. 
        traderData = jsonpickle.encode(star_state) 
        # Sample conversion request. Check more details below. 
        conversions = None
        return result, conversions, traderData