from datamodel import Trade, Listing, OrderDepth, UserId, TradingState, Order
from typing import List

def StarTrader(state: TradingState) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 2:
            return 9
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 2:
            return -9
        else:
            return -20
    # Order survival curve
    order_surv_curve = [1.   , 0.438, 0.295, 0.255, 0.213, 0.175, 0.157, 0.138, 0.116,
                        0.096, 0.075, 0.059, 0.046, 0.032, 0.018, 0.004, 0.003, 0.002,
                        0.002, 0.002, 0.001, 0.001, 0.001]
    
    symbol = "STARFRUIT"

    # Extract the relevant info from the trading_state
    sold_position = state.position.get(symbol,0)
    bought_position = state.position.get(symbol,0)
    bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
    ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))

    # Use the outermost prices as ref_price
    ref_price = (ask_book[-1][0] + bid_book[-1][0])/2.0

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

    # Placing limit bid orders
    front_in_Q = 0
    for idx, tmp in enumerate(bid_book):
        price, qty = tmp
        curr_profit = ref_price - (price+1)
        # Never do a -EV trade
        if curr_profit < 0:
            continue
        # If we are at the end of orderbook, there is no reason to do queue maximization        
        elif idx == len(bid_book)-1:
            placed_buy = buy_schedule(curr_profit) - bought_position
            if placed_buy >0:
                result.append(Order(symbol, price+1, placed_buy))
            front_in_Q += placed_buy + qty
            break
        else:
            # Do queue maximization
            placed_buy = 0
            # Calculate the maximum exposure allowed
            max_buy = buy_schedule(curr_profit) - bought_position
            # Calculate our position in the queue if we undercut the current bid
            # We compare it with the position and profit if we undercut the next bid
            curr_pos_in_Q = front_in_Q 
            next_pos_in_Q = front_in_Q + qty
            next_profit = ref_price - (bid_book[idx+1][0]+1)

            if next_pos_in_Q < len(order_surv_curve):
                # The difference in expected profit if we undercut the current bid vs the next bid
                profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit 
                # While this difference is positive, put more limit order to undercut the current bid
                while placed_buy < max_buy and next_pos_in_Q < len(order_surv_curve) and profit_delta >0:
                    curr_pos_in_Q += 1
                    next_pos_in_Q += 1
                    placed_buy += 1
                    if next_pos_in_Q < len(order_surv_curve):
                        profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit
                    else:
                        break  
                # If we stop the loop due to lack of data, then dump all the remaining allowed exposure to undercut this bid
                if next_pos_in_Q >= len(order_surv_curve):
                    placed_buy = max_buy
                if placed_buy >0:
                    result.append(Order(symbol, price+1, placed_buy))
                bought_position += placed_buy
                front_in_Q += placed_buy + qty
            else:
                if max_buy >0:
                    result.append(Order(symbol, price+1, max_buy))
                bought_position += max_buy
                front_in_Q += max_buy + qty

   # Placing limit ask orders
    front_in_Q = 0
    for idx, tmp in enumerate(ask_book):
        price, qty = tmp
        curr_profit = (price-1) - ref_price
        # Never do a -EV trade
        if curr_profit < 0:
            continue
        # If we are at the end of orderbook, there is no reason to do queue maximization        
        elif idx == len(ask_book)-1:
            placed_sell =  sold_position - sell_schedule(curr_profit)
            if placed_sell >0:
                result.append(Order(symbol, price-1, -placed_sell))
            front_in_Q += placed_sell + qty
            break
        else:
            # Do queue maximization
            placed_sell = 0
            # Calculate the maximum exposure allowed
            max_sell = sold_position-sell_schedule(curr_profit)
            # Calculate our position in the queue if we undercut the current bid
            # We compare it with the position and profit if we undercut the next bid
            curr_pos_in_Q = front_in_Q 
            next_pos_in_Q = front_in_Q + qty
            next_profit = (ask_book[idx+1][0]-1)-ref_price

            if next_pos_in_Q < len(order_surv_curve):
                # The difference in expected profit if we undercut the current bid vs the next bid
                profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit 
                # While this difference is positive, put more limit order to undercut the current bid
                while placed_sell < max_sell  and profit_delta >0:
                    curr_pos_in_Q += 1
                    next_pos_in_Q += 1
                    placed_sell += 1
                    if next_pos_in_Q < len(order_surv_curve):
                        profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit
                    else:
                        break 
                # If we stop the loop due to lack of data, then dump all the remaining allowed exposure to undercut this bid
                if next_pos_in_Q >= len(order_surv_curve):
                    placed_sell = max_sell
                if placed_sell >0:
                    result.append(Order(symbol, price-1, -placed_sell))
                sold_position -= placed_sell
                front_in_Q += placed_sell + qty
            else:
                if max_sell >0:
                    result.append(Order(symbol, price-1, -max_sell))
                sold_position -= max_sell
                front_in_Q += max_sell + qty

    return result

def AmetTrader(state: TradingState) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 3:
            return 17
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        elif x < 3:
            return -17
        else:
            return -20
    # Order survival curve
    order_surv_curve = [1.   , 0.451, 0.257, 0.199, 0.148, 0.102, 0.078, 
                        0.062, 0.04 ,0.031, 0.021, 0.011, 0.003, 0.002, 
                        0.002, 0.002, 0.001, 0.001]
    
    symbol = "AMETHYSTS"
 
    # Extract the relevant info from the trading_state
    sold_position = state.position.get(symbol,0)
    bought_position = state.position.get(symbol,0)
    bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
    ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))

    # Use the outermost prices as ref_price
    ref_price = (ask_book[-1][0] + bid_book[-1][0])/2.0

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

    # Placing limit bid orders
    front_in_Q = 0
    for idx, tmp in enumerate(bid_book):
        price, qty = tmp
        curr_profit = ref_price - (price+1)
        # Never do a -EV trade
        if curr_profit < 0:
            continue
        # If we are at the end of orderbook, there is no reason to do queue maximization        
        elif idx == len(bid_book)-1:
            placed_buy = buy_schedule(curr_profit) - bought_position
            if placed_buy >0:
                result.append(Order(symbol, price+1, placed_buy))
            front_in_Q += placed_buy + qty
            break
        else:
            # Do queue maximization
            placed_buy = 0
            # Calculate the maximum exposure allowed
            max_buy = buy_schedule(curr_profit) - bought_position
            # Calculate our position in the queue if we undercut the current bid
            # We compare it with the position and profit if we undercut the next bid
            curr_pos_in_Q = front_in_Q 
            next_pos_in_Q = front_in_Q + qty
            next_profit = ref_price - (bid_book[idx+1][0]+1)

            if next_pos_in_Q < len(order_surv_curve):
                # The difference in expected profit if we undercut the current bid vs the next bid
                profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit 
                # While this difference is positive, put more limit order to undercut the current bid
                while placed_buy < max_buy and next_pos_in_Q < len(order_surv_curve) and profit_delta >0:
                    curr_pos_in_Q += 1
                    next_pos_in_Q += 1
                    placed_buy += 1
                    if next_pos_in_Q < len(order_surv_curve):
                        profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit
                    else:
                        break  
                # If we stop the loop due to lack of data, then dump all the remaining allowed exposure to undercut this bid
                if next_pos_in_Q >= len(order_surv_curve):
                    placed_buy = max_buy
                if placed_buy >0:
                    result.append(Order(symbol, price+1, placed_buy))
                bought_position += placed_buy
                front_in_Q += placed_buy + qty
            else:
                if max_buy >0:
                    result.append(Order(symbol, price+1, max_buy))
                bought_position += max_buy
                front_in_Q += max_buy + qty

   # Placing limit ask orders
    front_in_Q = 0
    for idx, tmp in enumerate(ask_book):
        price, qty = tmp
        curr_profit = (price-1) - ref_price
        # Never do a -EV trade
        if curr_profit < 0:
            continue
        # If we are at the end of orderbook, there is no reason to do queue maximization        
        elif idx == len(ask_book)-1:
            placed_sell =  sold_position - sell_schedule(curr_profit)
            if placed_sell >0:
                result.append(Order(symbol, price-1, -placed_sell))
            front_in_Q += placed_sell + qty
            break
        else:
            # Do queue maximization
            placed_sell = 0
            # Calculate the maximum exposure allowed
            max_sell = sold_position-sell_schedule(curr_profit)
            # Calculate our position in the queue if we undercut the current bid
            # We compare it with the position and profit if we undercut the next bid
            curr_pos_in_Q = front_in_Q 
            next_pos_in_Q = front_in_Q + qty
            next_profit = (ask_book[idx+1][0]-1)-ref_price

            if next_pos_in_Q < len(order_surv_curve):
                # The difference in expected profit if we undercut the current bid vs the next bid
                profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit 
                # While this difference is positive, put more limit order to undercut the current bid
                while placed_sell < max_sell  and profit_delta >0:
                    curr_pos_in_Q += 1
                    next_pos_in_Q += 1
                    placed_sell += 1
                    if next_pos_in_Q < len(order_surv_curve):
                        profit_delta = order_surv_curve[curr_pos_in_Q]*curr_profit - order_surv_curve[next_pos_in_Q]*next_profit
                    else:
                        break 
                # If we stop the loop due to lack of data, then dump all the remaining allowed exposure to undercut this bid
                if next_pos_in_Q >= len(order_surv_curve):
                    placed_sell = max_sell
                if placed_sell >0:
                    result.append(Order(symbol, price-1, -placed_sell))
                sold_position -= placed_sell
                front_in_Q += placed_sell + qty
            else:
                if max_sell >0:
                    result.append(Order(symbol, price-1, -max_sell))
                sold_position -= max_sell
                front_in_Q += max_sell + qty

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