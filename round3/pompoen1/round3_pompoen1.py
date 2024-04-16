import math
import statistics as stat
from collections import *
from typing import *

from datamodel import *

POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
UNDERCUT_SPREAD = {'AMETHYSTS': 1, 'STARFRUIT': 1, 'GIFT_BASKET': 4}


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

# This one does hits stale orders in the markets
class GiftBasketTrader2(BaseTrader):
    DEFAULT_TRADER_DATA = {'ref_price': float, 'volume': int, 'inventory_loss': float, 'exposure': float}

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        result = {}
        conversions = 0

        # Create orders
        for product in ['GIFT_BASKET']:
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
            case 'GIFT_BASKET':
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
                case 'GIFT_BASKET':
                    return min(60, round(x * 10))

        # Maximum negative exposure that we want to have if the profit is x
        def sell_schedule(x: float) -> int:
            match product:
                case 'GIFT_BASKET':
                    return max(-60, round(-x * 10))

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

        return result

'''TRADER FOR THE GIFT BASKET'''
class GiftBasketTrader(BaseTrader):
    DEFAULT_TRADER_DATA = {'ref_price': float, 'volume': int, 'inventory_loss': float, 'exposure': float}

    def algo(self, state: TradingState, trader_data: dict) -> tuple[dict[Symbol, list[Order]], int]:
        books = {} # might be used for algorithm, not necessary perse
        for product in ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']:
            # Just logging, only for the visualizer
            if product not in state.listings.keys():
                self.logger.print('Missing product', product)
                return {}, 0

            # Save volume, only used for visualizer
            if product in state.own_trades:
                trader_data['volume'][product] += sum([trade.quantity for trade in state.own_trades[product] if trade.timestamp == state.timestamp - 100])

            # Compute reference price
            ref_price = self.compute_product_price(product, state, trader_data)

            # Log exposure gain/loss, only for visualizer
            if product in trader_data['ref_price']:
                trader_data['exposure'][product] += (ref_price - trader_data['ref_price'][product]) * state.position.get(product, 0)
            trader_data['ref_price'][product] = ref_price

            # Create books
            asks = OrderedDict[int, int](sorted(state.order_depths[product].sell_orders.items()))
            bids = OrderedDict[int, int](sorted(state.order_depths[product].buy_orders.items(), reverse=True))
            books[product] = (bids, asks) # might be used for algorithm, not necessary perse

        self.logger.print('GIFT_BASKET')
        result = self.compute_orders_old(books, state, trader_data)
        conversions = 0

        return result, conversions

    # Calculate mid price of product
    def compute_product_price(self, product: str, state: TradingState, trader_data: dict) -> float:
        bid_book = list(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        ask_book = list(sorted(state.order_depths[product].sell_orders.items()))
        if not bid_book or not ask_book:
            self.logger.print('EMPTY BOOK!!!')
            return trader_data['ref_price'][product] if product in trader_data['ref_price'] else None

        return (ask_book[-1][0] + bid_book[-1][0]) / 2
    

    # Basically copy paste from Stanford guys. Some differences, basket std, basket components
    def compute_orders_old(self, books: dict[str, tuple[OrderedDict, OrderedDict]], state: TradingState, trader_data: dict) -> dict[Symbol, list[Order]]:
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        orders = {'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}

        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            # Ordered buys and sells
            osell[p] = OrderedDict[int, int](sorted(state.order_depths[p].sell_orders.items()))
            obuy[p] = OrderedDict[int, int](sorted(state.order_depths[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0

            for price, vol in obuy[p].items():
                vol_buy[p] += vol
                if vol_buy[p] >= POSITION_LIMIT[p] / 10:
                    break

            for price, vol in osell[p].items():
                vol_sell[p] += -vol
                if vol_sell[p] >= POSITION_LIMIT[p] / 10:
                    break

        # spread between components and gift basket, when price deviates std 0.5 open a position
        res_quote = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE'] * 4 - mid_price['STRAWBERRIES'] * 6 - mid_price['ROSES'] - 380

        basket_std = 75
        trade_at = basket_std * 0.5

        pb_pos = state.position.get('GIFT_BASKET', 0)
        pb_neg = state.position.get('GIFT_BASKET', 0)

        # Sell basket if basket overpriced
        if res_quote > trade_at:
            vol_max_basket = state.position.get('GIFT_BASKET', 0) + POSITION_LIMIT['GIFT_BASKET']
            # vol_max_chocolate = POSITION_LIMIT['CHOCOLATE'] - state.position.get('CHOCOLATE', 0)
            # vol_max_strawberries = POSITION_LIMIT['STRAWBERRIES'] - state.position.get('STRAWBERRIES', 0)
            # vol_max_roses = POSITION_LIMIT['ROSES'] - state.position.get('ROSES', 0)
            
            if vol_max_basket > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol_max_basket))
                # orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol_max_chocolate))
                # orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol_max_strawberries))
                # orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol_max_roses))
                pb_neg -= vol_max_basket
                
        # Buy basket if basket underpriced
        elif res_quote < -trade_at:
            vol_max_basket = POSITION_LIMIT['GIFT_BASKET'] - state.position.get('GIFT_BASKET', 0)
            # vol_max_chocolate = state.position.get('CHOCOLATE', 0) + POSITION_LIMIT['CHOCOLATE']
            # vol_max_strawberries = state.position.get('STRAWBERRIES', 0) + POSITION_LIMIT['STRAWBERRIES']
            # vol_max_roses = state.position.get('ROSES', 0) + POSITION_LIMIT['ROSES']
            
            if vol_max_basket > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol_max_basket))
                # orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol_max_chocolate))
                # orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol_max_strawberries))
                # orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol_max_roses))
                pb_pos += vol_max_basket

        return orders

    def compute_orders(self, books: dict[str, tuple[OrderedDict, OrderedDict]], state: TradingState, trader_data: dict) -> dict[Symbol, list[Order]]:
        orders = defaultdict(lambda: defaultdict(int))

        def get_avg_quote(book: OrderedDict, requested_volume: int, update_book: bool = False) -> float:
            quotes = []
            for price, qnt in book.items():
                if len(quotes) < requested_volume and abs(qnt) > 0:
                    vol = min(requested_volume - len(quotes), abs(qnt))
                    for _ in range(vol):
                        quotes.append(price)
                        if update_book: book[price] += 1 if qnt < 0 else -1

                if len(quotes) == requested_volume: return stat.mean(quotes)

            return None

        def fill_qnt(book: OrderedDict, requested_volume: int, update_book: bool = True) -> List[tuple[int, int]]:
            orders = []
            filled_vol = 0
            for price, qnt in book.items():
                if len(orders) < requested_volume and abs(qnt) > 0:
                    vol = min(requested_volume - filled_vol, abs(qnt))
                    if vol > 0:
                        orders.append((price, vol if qnt < 0 else -vol))
                        if update_book: book[price] += vol if qnt < 0 else -vol
                        filled_vol += vol

            return orders

        for direction in range(2):
            while True:
                avg_ask_gift = get_avg_quote(books['GIFT_BASKET'][1 - direction], 1)
                avg_ask_choco = get_avg_quote(books['CHOCOLATE'][1 - direction], 4)
                avg_ask_straw = get_avg_quote(books['STRAWBERRIES'][1 - direction], 6)
                avg_ask_roses = get_avg_quote(books['ROSES'][1 - direction], 1)

                if avg_ask_gift is None or avg_ask_choco is None or avg_ask_straw is None or avg_ask_roses is None: break

                basket_nav_bid = 4 * avg_ask_choco + 6 * avg_ask_straw + avg_ask_roses + 380

                if abs(avg_ask_gift - basket_nav_bid) > 30 :
                    for price, vol in fill_qnt(books['GIFT_BASKET'][1 - direction], 1): orders['GIFT_BASKET'][price] += vol
                    for price, vol in fill_qnt(books['CHOCOLATE'][1 - direction], 4): orders['CHOCOLATE'][price] += vol
                    for price, vol in fill_qnt(books['STRAWBERRIES'][1 - direction], 6): orders['STRAWBERRIES'][price] += vol
                    for price, vol in fill_qnt(books['ROSES'][1 - direction], 1): orders['ROSES'][price] += vol

                else: break

        result = defaultdict(list)

        for product, product_orders in orders.items():
            for price, qnt in product_orders.items():
                result[product].append(Order(product, price, qnt))

        return result


class Trader:
    logger = Logger()
    TRADERS = [AmethistsStarfruitTrader(), OrchidsTrader(), GiftBasketTrader2(), GiftBasketTrader()]

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
