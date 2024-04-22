import math
import statistics as stat
from collections import *
from typing import *

from datamodel import *

POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20,
                  'ORCHIDS': 100,
                  'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60,
                  'COCONUT': 300, 'COCONUT_COUPON': 600}
UNDERCUT_SPREAD = {'AMETHYSTS': 1, 'STARFRUIT': 1}


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
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

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


'''TRADER BASE CLASS'''
class BaseTrader:
    BASE_TRADER_DATA = {'volume': int}
    TRADER_DATA = {}
    logger = None

    def with_logger(self, logger: Logger):
        self.logger = logger
        return self

    def run(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        for k, v in self.BASE_TRADER_DATA.items() | self.TRADER_DATA.items():
            trader_data[k] = defaultdict(v, trader_data[k]) if k in trader_data else defaultdict(v)

        for product, trades in state.own_trades.items():
            trader_data['volume'][product] += sum([trade.quantity for trade in trades if trade.timestamp == state.timestamp - 100])

        return self.algo(state, trader_data)

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        assert False, 'not implemented'


'''TRADER FOR STARFRUIT AND AMETHYSTS'''
class AmethistsStarfruitTrader(BaseTrader):
    TRADER_DATA = {'ref_price': float, 'inventory_loss': float, 'exposure': float}

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        result = {}
        conversions = 0

        # Create orders
        for product in ['STARFRUIT', 'AMETHYSTS']:
            if product not in state.listings.keys(): continue
            self.logger.print(product)

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
                    return min(20, round(x * 10))

        # Maximum negative exposure that we want to have if the profit is x
        def sell_schedule(x: float) -> int:
            match product:
                case 'AMETHYSTS':
                    return max(-20, round(-x * 10))
                case 'STARFRUIT':
                    return max(-20, round(-x * 10))

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
            quote = price + UNDERCUT_SPREAD[product] if price is not None else math.floor(ref_price) - UNDERCUT_SPREAD[product]
            self.logger.print('Incomplete buy book', [res for res in result if res.quantity > 0], quote)
            result.append(Order(product, quote, POSITION_LIMIT[product] - bought_position))

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
            quote = price - UNDERCUT_SPREAD[product] if price is not None else math.ceil(ref_price) + UNDERCUT_SPREAD[product]
            self.logger.print('Incomplete sell book', [res for res in result if res.quantity < 0], quote)
            result.append(Order(product, quote, -POSITION_LIMIT[product] - sold_position))

        return result


'''TRADER FOR ORCHIDS'''
class OrchidsTrader(BaseTrader):
    TRADER_DATA = {'exposure': float, 'ref_price': float}

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        result = {}
        conversions = 0

        for product in ['ORCHIDS']:
            if product not in state.listings.keys(): continue
            self.logger.print(product)

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
        # liquidity at -4 and -5, but it didn't increase the pnl. It might be related to other trades.

        if midprice >= (south_ask + south_bid) / 2:
            sold_position = 0
            reservation_bid = max(math.ceil(south_ask), math.floor(conversion_observation.askPrice) - 2)

            # Could be useless
            for price, qty in bid_book:
                arbitrage_profit = price - south_ask
                reservation_bid_profit = price - reservation_bid
                executed_sell = min(sold_position + POSITION_LIMIT[product], qty)

                if (arbitrage_profit > 4 or reservation_bid_profit > 0) and executed_sell > 0:
                    self.logger.print('arbitrage opportunity (price, qty, profit):', price, executed_sell, arbitrage_profit * executed_sell)
                    orders.append(Order(product, price, -executed_sell))
                    sold_position -= executed_sell

            orders.append(Order(product, reservation_bid, -POSITION_LIMIT[product] - sold_position))
            self.logger.print('market making (price, qty)', reservation_bid, -POSITION_LIMIT[product] - sold_position)

        else:
            bought_position = 0
            reservation_ask = min(math.floor(south_bid), math.ceil(conversion_observation.bidPrice) + 2)

            # Could be useless
            for price, qty in ask_book:
                arbitrage_profit = south_bid - price
                reservation_ask_profit = reservation_ask - price
                executed_buy = min(POSITION_LIMIT[product] - bought_position, -qty)

                if (arbitrage_profit > 4 or reservation_ask_profit > 0) and executed_buy > 0:
                    self.logger.print('arbitrage opportunity (price, qty, profit):', price, executed_buy, arbitrage_profit * executed_buy)
                    orders.append(Order(product, price, executed_buy))
                    bought_position += executed_buy

            orders.append(Order(product, reservation_ask, POSITION_LIMIT[product] - bought_position))
            self.logger.print('market making (price, qty)', reservation_ask, POSITION_LIMIT[product] - bought_position)

        return orders, conversions


'''TRADER FOR THE GIFT BASKET'''
class GiftBasketTrader(BaseTrader):
    TRADER_DATA = {'ref_price': float, 'exposure': float}

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        for product in ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']:
            if product not in state.listings.keys():
                self.logger.print('Missing product', product)
                return {}, 0

            # Compute reference price
            ref_price = self.compute_product_price(product, state, trader_data)

            # Log exposure gain/loss
            if product in trader_data['ref_price']:
                trader_data['exposure'][product] += (ref_price - trader_data['ref_price'][product]) * state.position.get(product, 0)
            trader_data['ref_price'][product] = ref_price

        self.logger.print('GIFT_BASKET')
        result = self.compute_orders(state, trader_data)
        conversions = 0

        return result, conversions

    def compute_product_price(self, product: str, state: TradingState, trader_data: dict) -> float:
        bid_book = list(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        ask_book = list(sorted(state.order_depths[product].sell_orders.items()))
        if not bid_book or not ask_book:
            self.logger.print('EMPTY BOOK!!!')
            return trader_data['ref_price'][product] if product in trader_data['ref_price'] else None

        return (ask_book[-1][0] + bid_book[-1][0]) / 2


    def compute_orders(self, state: TradingState, trader_data: dict) -> dict[Symbol, list[Order]]:
        orders = defaultdict(list)

        gift_basket = trader_data['ref_price']['GIFT_BASKET']
        chocolate = trader_data['ref_price']['CHOCOLATE']
        strawberries = trader_data['ref_price']['STRAWBERRIES']
        roses = trader_data['ref_price']['ROSES']

        basket_spread = gift_basket - (chocolate * 4 + strawberries * 6 + roses + 380)
        self.logger.print('basket_spread', basket_spread)

        basket_std = 76
        trade_at_to_sell = basket_std * 0.7
        trade_at_to_buy = basket_std * 0.5

        if basket_spread > trade_at_to_sell:
            vol = state.position.get('GIFT_BASKET', 0) + POSITION_LIMIT['GIFT_BASKET']
            worst_buy = min(state.order_depths['GIFT_BASKET'].buy_orders.keys())

            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy, -vol))

        elif basket_spread < -trade_at_to_buy:
            vol = POSITION_LIMIT['GIFT_BASKET'] - state.position.get('GIFT_BASKET', 0)
            worst_sell = max(state.order_depths['GIFT_BASKET'].sell_orders.keys())

            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell, vol))

        return orders


'''TRADER FOR COCONUT'''
class CoconutTrader(BaseTrader):
    TRADER_DATA = {'ref_price': float, 'exposure': float,"coco_spread_state" : float} # 0 : neutral, 1: long, -1 : short

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        for product in ['COCONUT', "COCONUT_COUPON"]:
            if product not in state.listings.keys():
                self.logger.print('Missing product', product)
                return {}, 0

            # Compute reference price
            ref_price = self.compute_product_price(product, state, trader_data)

            # Log exposure gain/loss
            if product in trader_data['ref_price']:
                trader_data['exposure'][product] += (ref_price - trader_data['ref_price'][product]) * state.position.get(product, 0)
            trader_data['ref_price'][product] = ref_price

        self.logger.print('coconut trader')
        result = self.compute_orders(state, trader_data)
        conversions = 0

        return result, conversions

    def compute_product_price(self, product: str, state: TradingState, trader_data: dict) -> float:
        bid_book = list(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        ask_book = list(sorted(state.order_depths[product].sell_orders.items()))
        if not bid_book or not ask_book:
            self.logger.print('EMPTY BOOK!!!')
            return trader_data['ref_price'][product] if product in trader_data['ref_price'] else None

        return (ask_book[-1][0] + bid_book[-1][0]) / 2


    def compute_orders(self, state: TradingState, trader_data: dict) -> dict[Symbol, list[Order]]:
        orders = defaultdict(list)

        coconut = trader_data['ref_price']['COCONUT']
        coupon = trader_data['ref_price']['COCONUT_COUPON']

        spread = coupon - 0.5*coconut + 4380 
        self.logger.print('coconut_spread', spread)
        
        # If the spread is larger than pos_threshold, then go short, otherwise go long
        pos_threshold = 6
        neg_threshold = -5
        if spread >= pos_threshold:
            trader_data['coco_spread_state']["state"] = -1.0
        elif spread <= neg_threshold:
            trader_data['coco_spread_state']["state"] = 1.0
        spread_state = trader_data["coco_spread_state"].get("state",0)
        if spread_state  > 0.9: # Go long
            coupon_pos = state.position.get('COCONUT_COUPON', 0) 
            coupon_buy_vol = POSITION_LIMIT['COCONUT_COUPON'] - coupon_pos # The position limit coincidentally gives us the correct hedging ratio
            tmp = state.order_depths['COCONUT_COUPON'].sell_orders.keys()
            if len(tmp) > 0:
                cheapest_coupon = min(tmp)
                if coupon_buy_vol > 0:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', cheapest_coupon, coupon_buy_vol))


            coconut_pos = state.position.get('COCONUT', 0) 
            coconut_sell_vol = coconut_pos + POSITION_LIMIT['COCONUT']
            tmp = state.order_depths['COCONUT'].buy_orders.keys()
            if len(tmp) >0:
                richest_coconut = max(tmp)
                if coconut_sell_vol > 0:
                    orders['COCONUT'].append(Order('COCONUT', richest_coconut, -coconut_sell_vol))

        elif spread_state < -0.9:
            coupon_pos = state.position.get('COCONUT_COUPON', 0) 
            coupon_sell_vol = coupon_pos +  POSITION_LIMIT['COCONUT_COUPON']  
            tmp = state.order_depths['COCONUT_COUPON'].buy_orders.keys()
            if len(tmp) >0:
                richest_coupon = max(tmp)
                if coupon_sell_vol > 0:
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', richest_coupon, -coupon_sell_vol))


            coconut_pos = state.position.get('COCONUT', 0) 
            coconut_buy_vol =  POSITION_LIMIT['COCONUT'] - coconut_pos
            tmp = state.order_depths['COCONUT'].sell_orders.keys()
            if len(tmp) >0:
                cheapest_coconut = min(tmp)
                if coconut_buy_vol > 0:
                    orders['COCONUT'].append(Order('COCONUT', cheapest_coconut, coconut_buy_vol))
        return orders


class Trader:
    logger = Logger()
    #TRADERS = [AmethistsStarfruitTrader(), OrchidsTrader(), GiftBasketTrader(), CoconutTrader()]
    TRADERS = [CoconutTrader()]

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
