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

def StarTrader(state: TradingState) -> List[Order]:
    # Maximum positive exposure that we want to have if the profit is x
    def buy_schedule(x : float) -> int:
        if x <= 0:
            return 0
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        else:
            return -20
    
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
        else:
            return 20
    # Maximum negative exposure that we want to have if the profit is x
    def sell_schedule(x : float) -> int:
        if x <= 0:
            return 0
        else:
            return -20
    
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
        logger = Logger(local=False)

        result["STARFRUIT"]= StarTrader(state)
        result["AMETHYSTS"]= AmetTrader(state)

        # String value holding Trader state data required. 
        traderData = ""
        # Sample conversion request. Check more details below. 
        conversions = None
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData