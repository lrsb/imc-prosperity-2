'''PARAMS'''
MAX_TRADER_DATA_LEN = 4

# Product limits
POSITION_LIMIT = { 'STARFRUIT': 20, 'AMETHYSTS': 20 }

# Traders params
LIN_REG = { 'STARFRUIT': [0.5594, 0.2365, 0.1156, 0.0885] }
QUOTES_SPREAD = { 'STARFRUIT': 1, 'AMETHYSTS': 1 }
RISK_ADVERSION = { 'STARFRUIT': 1, 'AMETHYSTS': 2 }

'''IMPORTS'''
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


'''LOGGING'''
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


'''TRADER BASE CLASS'''
class BaseTrader:
    DEFAULT_TRADER_DATA = {}
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
    DEFAULT_TRADER_DATA = { 'fair_price': int, 'volume': int, 'inventory_loss': float, 'exposure': float, 'midprices': list }

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        result = {}

        # Create orders
        for product in ['STARFRUIT', 'AMETHYSTS']:
            # Clean old data
            if len(trader_data['midprices'][product]) >= MAX_TRADER_DATA_LEN:
                trader_data['midprices'][product].pop(0)

            # Calculate volume weighted mid price
            book = state.order_depths[product]
            sell = stat.fmean([price for price, _ in book.sell_orders.items()], weights=[qnt for _, qnt in book.sell_orders.items()])
            buy = stat.fmean([price for price, _ in book.buy_orders.items()], weights=[qnt for _, qnt in book.buy_orders.items()])
            if book.sell_orders and book.sell_orders:
                trader_data['midprices'][product].append((buy + sell) / 2)
            else:
                trader_data['midprices'][product].append(trader_data['midprices'][product][-1])
            self.logger.print(product, 'vwmp', buy, sell)

            # Save volume
            if product in state.own_trades:
                trader_data['volume'][product] += sum([trade.quantity for trade in state.own_trades[product] if trade.timestamp == state.timestamp - 100])

            # Compute fair mid price
            midprice = self.compute_product_price(product, trader_data)

            # Log exposure gain/loss
            if product in trader_data['fair_price']:
                trader_data['exposure'][product] += (midprice - trader_data['fair_price'][product]) * state.position.get(product, 0)
            trader_data['fair_price'][product] = midprice

            # Compute orders
            result[product] = self.compute_orders(product, state, trader_data)

            # Clean STARFRUIT orders if regression is not available
            if product == 'STARFRUIT' and len(trader_data['midprices'][product]) < len(LIN_REG[product]): del result[product]

        return result, 0


    def compute_product_price(self, product: str, trader_data: dict) -> float:
        computed_mid_price = None

        match product:
            case 'AMETHYSTS':
                computed_mid_price = 10000

            case 'STARFRUIT':
                def lin_reg(data):
                    coeff = LIN_REG[product]
                    return sum([coeff[i] * e for i, e in enumerate(data[::-1][-len(coeff):])])

                computed_mid_price = lin_reg(trader_data['midprices'][product])

                if len(trader_data['midprices'][product]) < len(LIN_REG[product]):
                    computed_mid_price /= sum(LIN_REG[product][:len(trader_data['midprices'][product])])

        return computed_mid_price


    def compute_orders(self, product: str, state: TradingState, trader_data: dict) -> List[Order]:
        orders = []
        asks = OrderedDict[int, int](sorted(state.order_depths[product].sell_orders.items()))
        bids = OrderedDict[int, int](sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        midprice = trader_data['fair_price'][product]

        # Calculate trade limits
        inventory_limit = POSITION_LIMIT[product]
        inventory_current = state.position.get(product, 0)

        max_buy = inventory_limit - inventory_current
        max_sell = -inventory_limit - inventory_current

        assert max_buy >= 0
        assert max_sell <= 0

        # Use a dynamic spread to better manage inventory
        inventory_spread = RISK_ADVERSION[product] * inventory_current / inventory_limit
        # If inventory is positive we increase bid-mid spread, always positive
        buy_spread = max(inventory_spread, 0)
        # If inventory is negative we increase ask-mid spread, always positive
        sell_spread = -min(inventory_spread, 0)

        bid, ask = midprice - buy_spread, midprice + sell_spread

        self.logger.print(product, 'quotes', bid, ask)

        # Match sell orders first, if the trade is convenient
        for price, qnt in asks.items():
            if price <= bid and max_buy > 0:
                size = min(-qnt, max_buy)
                asks[price] += size
                orders.append(Order(product, price, size))

                max_buy -= size
                assert max_buy >= 0

        for price, qnt in asks.items():
            if price < math.floor(midprice) and qnt:
                trader_data['inventory_loss'][product] += abs((price - math.floor(midprice)) * qnt)
                assert trader_data['inventory_loss'][product] >= 0

        # Match buy orders first, if the trade is convenient
        for price, qnt in bids.items():
            if price >= ask and max_sell < 0:
                size = min(qnt, -max_sell)
                bids[price] -= size
                orders.append(Order(product, price, -size))

                max_sell += size
                assert max_sell <= 0

        for price, qnt in bids.items():
            if price > math.ceil(midprice) and qnt:
                trader_data['inventory_loss'][product] += abs((price - math.ceil(midprice)) * qnt)
                assert trader_data['inventory_loss'][product] >= 0

        # TODO instead of providing quotes better than market, place them near midprice based on historical bot volumes.
        # there are trades happening even if we provided quotes

        # And then provide a buy quote for bots above best feasible bid (highest bid that is less than or equal to our bid)
        if max_buy:
            placed = False
            for price, qnt in bids.items():
                if qnt and price + QUOTES_SPREAD[product] <= bid:
                    orders.append(Order(product, price + QUOTES_SPREAD[product], max_buy))
                    placed = True
                    break

            if not placed: orders.append(Order(product, math.floor(bid), max_buy))

        # And then provide a sell quote for bots below best feasible ask (smallest ask that is greater than or equal to our ask)
        if max_sell:
            placed = False
            for price, qnt in asks.items():
                if qnt and price - QUOTES_SPREAD[product] >= ask:
                    orders.append(Order(product, price - QUOTES_SPREAD[product], max_sell))
                    placed = True
                    break

            if not placed: orders.append(Order(product, math.ceil(ask), max_sell))

        return orders


'''TRADER'''
class Trader:
    logger = Logger(local=False)
    TRADERS = [AmethistsStarfruitTrader()]

    def local(self): self.logger = Logger(local=True)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Load or create trader data
        trader_data = json.loads(state.traderData) if state.traderData else {}
        # Reduce logs size state.traderData = ''

        result = {}
        conversions = 0

        for trader in self.TRADERS:
            trader_result, trader_conversions = trader.with_logger(self.logger).run(state, trader_data)

            result |= trader_result
            conversions += trader_conversions

        # Save trader data (pretty print for visualizer) and logs
        trader_data_json = json.dumps(trader_data, indent=2)
        self.logger.flush(state, result, conversions, trader_data_json)

        # Check order limits
        for product, orders in result.items():
            assert sum([order.quantity for order in orders if order.quantity > 0]) + state.position.get(product, 0) <= POSITION_LIMIT[product]
            assert sum([order.quantity for order in orders if order.quantity < 0]) + state.position.get(product, 0) >= -POSITION_LIMIT[product]

        return result, conversions, trader_data_json
