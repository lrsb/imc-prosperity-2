from datamodel import *
from typing import *
from collections import *
import statistics as stat
import numpy as np
import pandas as pd
import json
import math
import copy
import jsonpickle

POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100}
UNDERCUT_SPREAD = {'AMETHYSTS': 1, 'STARFRUIT': 1}

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

        if self.local:
            self.local_logs[state.timestamp] = output
        else:
            print(output)

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


'''TRADER BASE CLASS'''
class BaseTrader:
    DEFAULT_TRADER_DATA = {}
    logger = None

    def with_logger(self, logger: Logger):
        self.logger = logger
        return self

    def run(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        for k, v in self.DEFAULT_TRADER_DATA.items():
            trader_data[k] = defaultdict(v, trader_data[k]) if k in trader_data else defaultdict(v)

        return self.algo(state, trader_data)

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        assert False, 'not implemented'


'''TRADER FOR STARFRUIT AND AMETHYSTS'''
class AmethistsStarfruitTrader(BaseTrader):
    DEFAULT_TRADER_DATA = {'ref_price': float, 'volume': int, 'inventory_loss': float, 'exposure': float}

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        result = {}
        conversions = 0

        # Create orders
        for product in ['STARFRUIT', 'AMETHYSTS']:
            if product not in state.listings.keys(): continue
            self.logger.print(product)

            # Save volume
            if product in state.own_trades:
                trader_data['volume'][product] += sum([trade.quantity for trade in state.own_trades[product] if trade.timestamp == state.timestamp - 100])

            # Compute reference price
            ref_price = self.compute_product_price(product, state, trader_data)

            # Log exposure gain/loss
            if product in trader_data['ref_price']:
                trader_data['exposure'][product] += (ref_price - trader_data['ref_price'][product]) * state.position.get(product, 0)
            trader_data['ref_price'][product] = ref_price

            # Compute orders
            if ref_price is None: continue
            result[product] = self.compute_orders(product, ref_price, state, trader_data)

        return result, conversions

    def compute_product_price(self, product: str, state: TradingState, trader_data: dict) -> float:
        match product:
            case 'AMETHYSTS':
                return 10000

            case 'STARFRUIT':
                bid_book = list(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
                ask_book = list(sorted(state.order_depths[product].sell_orders.items()))
                if not bid_book or not ask_book:
                    self.logger.print('EMPTY BOOK!!!')
                    return trader_data['ref_price'][product] if product in trader_data['ref_price'] else None

                return (ask_book[-1][0] + bid_book[-1][0]) / 2

    def compute_orders(self, product: str, ref_price: float, state: TradingState, trader_data: dict) -> List[Order]:
        # Maximum positive exposure that we want to have if the profit is x
        def buy_schedule(x: float) -> int:
            match product:
                case 'AMETHYSTS':
                    return min(20, round(x * 10))
                case 'STARFRUIT':
                    return min(20, round(x * 8))

        # Maximum negative exposure that we want to have if the profit is x
        def sell_schedule(x: float) -> int:
            match product:
                case 'AMETHYSTS':
                    return max(-20, round(-x * 10))
                case 'STARFRUIT':
                    return max(-20, round(-x * 8))

        # Extract the relevant info from the trading_state
        sold_position = state.position.get(product, 0)
        bought_position = state.position.get(product, 0)

        bid_book = list(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        ask_book = list(sorted(state.order_depths[product].sell_orders.items()))

        # Placeholder for the emitted orders
        result = []

        # Removing stale orders
        n_ask_book = []
        for price, qty in ask_book:
            curr_profit = ref_price - price
            # we never take -EV trade, and the max positive exposure that we want to
            # hold depends on the buy_schedule
            executed_buy = min(buy_schedule(curr_profit) - bought_position, -qty)
            if curr_profit >= 0 and executed_buy > 0:
                result.append(Order(product, price, executed_buy))
                bought_position += executed_buy
                if executed_buy != -qty:
                    n_ask_book.append((price, qty + executed_buy))
            else:
                n_ask_book.append((price, qty))
                if curr_profit > 0:
                    trader_data['inventory_loss'][product] += abs(curr_profit * qty)
                    assert trader_data['inventory_loss'][product] >= 0

        ask_book = n_ask_book

        n_bid_book = []
        for price, qty in bid_book:
            curr_profit = price - ref_price
            # we never take -EV trade, and the max negative exposure that we want to
            # hold depends on the sell_schedule
            executed_sell = min(sold_position - sell_schedule(curr_profit), qty)
            if curr_profit >= 0 and executed_sell > 0:
                result.append(Order(product, price, -executed_sell))
                sold_position -= executed_sell
                if executed_sell != qty:
                    n_bid_book.append((price, qty - executed_sell))
            else:
                n_bid_book.append((price, qty))
                if curr_profit > 0:
                    trader_data['inventory_loss'][product] += abs(curr_profit * qty)
                    assert trader_data['inventory_loss'][product] >= 0

        bid_book = n_bid_book

        price = None
        # Placing limit orders
        for price, qty in bid_book:
            curr_profit = ref_price - (price + UNDERCUT_SPREAD[product])
            # we never take -EV trade, and the max positive exposure that we want to
            # hold depends on the buy_schedule
            placed_buy = buy_schedule(curr_profit) - bought_position
            if curr_profit >= 0 and placed_buy > 0:
                result.append(Order(product, price + UNDERCUT_SPREAD[product], placed_buy))
                bought_position += placed_buy

        if POSITION_LIMIT[product] > bought_position:
            self.logger.print('Incomplete buy book', [res for res in result if res.quantity > 0])
            result.append(Order(product, price if price is not None else math.floor(ref_price - 1), POSITION_LIMIT[product] - bought_position))

        price = None
        for price, qty in ask_book:
            curr_profit = (price - UNDERCUT_SPREAD[product]) - ref_price
            # we never take -EV trade, and the max negative exposure that we want to
            # hold depends on the sell_schedule
            placed_sell = sold_position - sell_schedule(curr_profit)
            if curr_profit >= 0 and placed_sell > 0:
                result.append(Order(product, price - UNDERCUT_SPREAD[product], -placed_sell))
                sold_position -= placed_sell

        if POSITION_LIMIT[product] > -sold_position:
            self.logger.print('Incomplete sell book', [res for res in result if res.quantity < 0])
            result.append(Order(product, price if price is not None else math.ceil(ref_price + 1), -POSITION_LIMIT[product] - sold_position))

        return result


'''TRADER FOR ORCHIDS'''
class OrchidsTrader(BaseTrader):
    DEFAULT_TRADER_DATA = {'volume': int, 'exposure': float, 'ref_price': float}

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        result = {}
        conversions = 0

        for product in ['ORCHIDS']:
            if product not in state.listings.keys(): continue
            self.logger.print(product)

            # Save volume
            if product in state.own_trades:
                trader_data['volume'][product] += sum([trade.quantity for trade in state.own_trades[product] if trade.timestamp == state.timestamp - 100])

            # Log exposure gain/loss
            conversion_observation = state.observations.conversionObservations[product]

            ref_price = (conversion_observation.askPrice + conversion_observation.bidPrice) / 2
            if product in trader_data['ref_price']:
                trader_data['exposure'][product] += (ref_price - trader_data['ref_price'][product]) * state.position.get(product, 0)
            trader_data['ref_price'][product] = ref_price

            # Generate orders and conversions
            product_orders, product_conversions = self.compute_orders_conversions(product, state, conversion_observation)
            result[product] = product_orders
            conversions += product_conversions

        return result, conversions

    def compute_orders_conversions(self, product: str, state: TradingState, conversion_observation: ConversionObservation) -> tuple[List[Order], int]:
        ask_book = list(sorted(state.order_depths[product].sell_orders.items()))
        bid_book = list(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        midprice = (max(state.order_depths[product].sell_orders.keys()) + min(state.order_depths[product].buy_orders.keys())) / 2

        south_ask = conversion_observation.askPrice + conversion_observation.importTariff + conversion_observation.transportFees
        south_bid = conversion_observation.bidPrice - conversion_observation.exportTariff - conversion_observation.transportFees
        self.logger.print('south island quotes', south_bid, south_ask)

        conversions = -state.position.get(product, 0)
        orders = []

        # The bots reservation price is always at -2, -4, -5 from south island ask price. I tried to catch the
        # liquidity at -4 and -5, but it didn't increase the pnl

        # TODO find a way to catch such liquidity

        if midprice > (south_ask + south_bid) / 2:
            sold_position = 0
            true_ask = math.floor(conversion_observation.askPrice) - 2

            for price, qty in bid_book:
                arbitrage_profit = price - true_ask
                executed_sell = min(sold_position + POSITION_LIMIT[product], qty)

                if arbitrage_profit > 0 and executed_sell > 0:
                    self.logger.print('arbitrage opportunity (price, qty, profit):', price, executed_sell, arbitrage_profit * executed_sell)
                    orders.append(Order(product, price, -executed_sell))
                    sold_position -= executed_sell

            orders.append(Order(product, true_ask, -POSITION_LIMIT[product] - sold_position))
            self.logger.print('market making (price, qty)', true_ask, -POSITION_LIMIT[product] - sold_position)

        else:
            bought_position = 0
            true_bid = math.ceil(conversion_observation.bidPrice) + 2

            for price, qty in ask_book:
                arbitrage_profit = true_bid - price
                executed_buy = min(POSITION_LIMIT[product] - bought_position, -qty)

                if arbitrage_profit > 0 and executed_buy > 0:
                    self.logger.print('arbitrage opportunity (price, qty, profit):', price, executed_buy, arbitrage_profit * executed_buy)
                    orders.append(Order(product, price, executed_buy))
                    bought_position += executed_buy

            orders.append(Order(product, true_bid, POSITION_LIMIT[product] - bought_position))
            self.logger.print('market making (price, qty)', true_bid, POSITION_LIMIT[product] - bought_position)

        return orders, conversions


class Trader:
    logger = Logger(local=False)
    TRADERS = [AmethistsStarfruitTrader(), OrchidsTrader()]
    #TRADERS = [AmethistsStarfruitTrader()]
    #TRADERS = [OrchidsTrader()]

    def local(self):
        self.logger = Logger(local=True)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Load or create trader data
        trader_data = json.loads(state.traderData) if state.traderData else {}

        result = {}
        conversions = 0

        for trader in self.TRADERS:
            trader_result, trader_conversions = trader.with_logger(self.logger).run(state, trader_data)

            result |= trader_result
            conversions += trader_conversions

        # Save trader data (pretty print for visualizer) and logs
        trader_data_json = json.dumps(trader_data, indent=2)
        self.logger.flush(state, result, conversions, trader_data_json)

        return result, conversions, trader_data_json
