from datamodel import *
from typing import List, Any


class Logger:
    local: bool
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        output = json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":"))

        if self.local: self.local_logs[state.timestamp] = output
        else: print(output)

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

def StarTrader(state: TradingState, trader_data: dict) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        return min(20, round(x * 10))
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        return max(-20, round(-x * 10))
    
    symbol = "STARFRUIT"



 
    # Extract the relevant info from the trading_state
    sold_position = state.position.get(symbol,0)
    bought_position = state.position.get(symbol,0)
    bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
    ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))

    # Use the outermost prices as ref_price
    ref_price = (ask_book[-1][0] + bid_book[-1][0])/2.0

    position_goal = 0
    price_change = trader_data[symbol]['prices'][-1] - trader_data[symbol]['prices'][0]
    if len(trader_data[symbol]['prices']) >= 20 and price_change >= 3:
        # ref_price += 1
        position_goal = 3
    # elif len(trader_data[symbol]['prices']) >= 20 and price_change <= -3:
    #     position_goal = -3
        # ref_price -= 1

    # Placeholder for the emitted orders
    result = []

    # Removing stale orders
    n_ask_book = []
    for price, qty in ask_book:
        curr_profit = ref_price - price
        # we never take -EV trade, and the max positive exposure that we want to
        # hold depends on the buy_schedule
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position - position_goal:
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
        if curr_profit >= 0 and sell_schedule(curr_profit) < sold_position + position_goal:
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
        if curr_profit >= 0 and buy_schedule(curr_profit) > bought_position - position_goal:
            placed_buy = buy_schedule(curr_profit) - bought_position
            result.append(Order(symbol, price+1, placed_buy))
            bought_position += placed_buy

    for price, qty in ask_book:
        curr_profit =  (price-1) - ref_price
        # we never take -EV trade, and the max negative exposure that we want to
        # hold depends on the sell_schedule
        if curr_profit >= 0 and sell_schedule(curr_profit) < sold_position + position_goal:
            placed_sell = sold_position-sell_schedule(curr_profit)
            result.append(Order(symbol, price-1, -placed_sell))
            sold_position -= placed_sell
    return result

def AmetTrader(state: TradingState, trader_data: dict) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        return min(20, round(x * 10))
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        return max(-20, round(-x * 10))
    
    symbol = "AMETHYSTS"
 
    # Extract the relevant info from the trading_state
    sold_position = state.position.get(symbol,0)
    bought_position = state.position.get(symbol,0)
    bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
    ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))

    # Use the outermost prices as ref_price
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
        logger = Logger(local=False)
        result = {}

        trader_data = self.guess_trend_direction(state)
        
        result["STARFRUIT"]= StarTrader(state, trader_data)
        result["AMETHYSTS"]= AmetTrader(state, trader_data)
        
        # Sample conversion request. Check more details below. 
        conversions = None
        logger.flush(state, result, conversions, jsonpickle.dumps(trader_data))
        return result, conversions, jsonpickle.dumps(trader_data)
    

    # Guesses the trend direction for any product given as input, but this is only used in the STARFRUIT trader
    def guess_trend_direction(self, state: TradingState):
        # String value holding Trader state data required. 
        if state.traderData == '' or state.traderData == None:
            trader_data = {}
            for symbol in state.order_depths:
                trader_data[symbol] = {'prices': [ ( max( state.order_depths[symbol].buy_orders ) + min( state.order_depths[symbol].sell_orders ) ) / 2 ]}
        else:
            trader_data = jsonpickle.loads(state.traderData)


        for symbol in state.order_depths:
            bid_book = list(sorted(state.order_depths[symbol].buy_orders.items(), reverse=True))
            ask_book = list(sorted(state.order_depths[symbol].sell_orders.items()))

            # Use the outermost prices as ref_price
            fair_price_this_run = (ask_book[-1][0] + bid_book[-1][0])/2.0
            prices = trader_data[symbol]['prices']
            prices.append( fair_price_this_run )

            ma_length = 100

            if len(prices) > ma_length:
                prices = prices[-ma_length:]

            average_fair_price = sum(prices) / len(prices)

            trader_data[symbol]['prices'] = prices
            trader_data[symbol]['avg price'] = average_fair_price

        return trader_data