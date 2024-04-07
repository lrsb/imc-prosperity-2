from typing import Dict, List
import json
import numpy as np
from copy import deepcopy
from datamodel import  TradingState, Order

class Configuration:
    def __init__(self):
        self.banana_mm_spread = (0.0,24.0)
        self.pearl_mm_spread = (5.0,5.0)
        
        self.pina_mm_vol_diff = 0.0
        self.pina_mm_spread_beta = 0.1353
        self.coco_mm_spread_beta = 0
        self.coco_mm_vol_diff = 0.01
        self.pina_mm_spread = 0.01
        self.coco_mm_spread = 0.01
        self.pinacoco_kf_Q1 = 8.907
        self.pinacoco_kf_Q2 = 4.647
        
        self.gear_mm_long = 4.403
        self.gear_mm_short = 9.102
        self.gear_mm_spread = 0.01158
        self.gear_mm_obs_bias = 0.1965
        self.gear_mm_price_bias = -0.06058


        # no lower bound
        # self.baguette_mm_spread = 0.07686377577350104
        # self.ukulele_mm_spread = 2.437357227485091
        # self.dip_mm_spread = 0.06970312360196539
        # self.basket_mm_spread = 0.6197530101920493
        # self.basket_mm_bias = 0.0772168438822171

        # all lower bound
        # self.baguette_mm_spread = 0.52882375
        # self.ukulele_mm_spread = 2.078197
        # self.dip_mm_spread = 0.30509362
        # self.basket_mm_spread = 0.86742
        # self.basket_mm_bias = 0.1234915617

        # # dip lower bound
        # self.baguette_mm_spread = 0.12799
        # self.ukulele_mm_spread = 2.66095
        # self.dip_mm_spread = 0.5257
        # self.basket_mm_spread = 0.99896
        # self.basket_mm_bias = 0.11091

        # #baguette lower bound
        # self.baguette_mm_spread=  0.8923748387618574
        # self.ukulele_mm_spread = 0.6335188094996463
        # self.dip_mm_spread = 1.690823245329407
        # self.basket_mm_spread = 0.46039361695889414
        # self.basket_mm_bias = 0.29722395977844296
         # 'config.basket_mm_ewma_length': 43, 'config.init_basket_price': 415

         #baguette lower bound 2
        self.baguette_mm_spread=  0.74265626
        self.ukulele_mm_spread = 2.943269
        self.dip_mm_spread = 0.2335144
        self.basket_mm_spread = 0.44757
        self.basket_mm_bias = 0.0550

        self.basket_mm_ewma_length = 56
        self.init_basket_price = 401

        
        self.banana_mm_survival_curve =[1.0, 0.37, 0.21, 0.185, 0.163, 0.145, 0.133, 0.114, 0.0982, 0.0838, 0.0726, 0.061,0.0478, 0.0397, 0.0331, 0.02843,0.0235, 0.0189, 0.0133, 0.00671,0.00075]
        self.pearl_mm_survival_curve = [1.0, 0.513, 0.16126, 0.1174591, 0.09, 0.0723, 0.0568, 0.0449, 0.0292, 0.019, 0.012646, 0.006,0.00139, 0.0011, 0.000834, 0.000834,0.00058, 0.000282, 0, 0,0]
        self.pina_mm_survival_curve = [1.        , 0.99688474, 0.82122495, 0.63927585, 0.46064774,
       0.31658144, 0.28836772, 0.25072004, 0.22565097, 0.19902428,
       0.16146476, 0.1270499 , 0.10665374, 0.06271675, 0.04076295,
       0.01569388, 0.01099159, 0.00628931, 0.00314465, 0.        ,
       0.        ]
        self.coco_mm_survival_curve = [1.        , 0.98834851, 0.97934865, 0.97165086, 0.95983721,
       0.87141138, 0.76805667, 0.67636725, 0.58396791, 0.47921021,
       0.35199571, 0.32772575, 0.31477218, 0.29373523, 0.27258357,
       0.25302468, 0.23364178, 0.21908162, 0.19931312, 0.17760293,
       0.15602127, 0.14012539, 0.12460128, 0.10685556, 0.08885582,
       0.0810295 , 0.06302977, 0.05385392, 0.04446845, 0.02717559,
       0.0130821 , 0.00925693, 0.00925693, 0.00803593, 0.0039537 ,
       0.0039537 , 0.0039537 , 0.0039537 , 0.        , 0.        ,
       0.        ]
        self.berry_mm_survival_curve = [1.        , 0.9375, 0.89583333, 0.82986111, 0.78125,
       0.69097222, 0.63541667, 0.57986111, 0.55208333, 0.51041667,
       0.38194444, 0.29166667, 0.20833333, 0.14930556, 0.10763889,
       0.06597222, 0.05902778, 0.04861111, 0.03472222, 0.01388889,
       0.00694444, 0.00694444, 0.00694444, 0.00694444, 0.00694444,
       0.00694444, 0.00694444, 0.00347222, 0.00347222, 0.00347222,
       0.00347222, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        ]


        self.gear_mm_survival_curve = [1.        , 0.65987168, 0.32348939, 0.00496867, 0.00248912,
       0.00100816, 0.        , 0.        , 0.        ]

        self.dip_mm_survival_curve = [1.        , 1.        , 1.        , 0.99753086, 0.8617284 ,
       0.68148148, 0.54814815, 0.41728395, 0.28148148, 0.12345679,
       0.        ]

        self.ukulele_mm_survival_curve = [1.        , 0.84960938, 0.64257812, 0.453125  , 0.27148438,
       0.07226562, 0.06054688, 0.0546875 , 0.04492188, 0.03515625,
       0.01367188, 0.00195312, 0.        , 0.        , 0.        ,
       0.        ]

        self.baguette_mm_survival_curve = [1.        , 1.        , 1.        , 0.84289277, 0.65835411,
       0.50124688, 0.3117207 , 0.14713217, 0.        , 0.        ,
       0.        ]

        self.basket_mm_survival_curve = [1.        , 0.66091954, 0.36206897, 0.32183908, 0.3045977 ,
       0.24712644, 0.18390805, 0.13793103, 0.06896552, 0.04022989,
       0.        ]


        self.logger_monitored_symbols = ['BANANAS', 'PEARLS', "PINA_COLADAS", "COCONUTS", 'BERRIES', 'DIVING_GEAR','DOLPHIN_SIGHTINGS', 'DIP', 'BAGUETTE','UKULELE','PICNIC_BASKET']
        # self.logger_monitored_symbols = ['BANANAS','PINA_COLADAS','COCONUTS','DIVING_GEAR','BERRIES']
        self.traders_tracked = ['Paris', 'Caesar', 'Camilla', 'Charlie', 'Pablo', 'Penelope', 'Gary', 'Peter', 'Gina' , 'Olivia']

        # self.traders_tracked = ['Olivia']


        self.tracking_coeffs = {}

        for trader in self.traders_tracked:
            for symbol in self.logger_monitored_symbols:
                self.tracking_coeffs[(trader,symbol)] = 0

        self.tracking_coeffs[('Paris','BANANAS')] = 0.1210
        self.tracking_coeffs[('Paris','DIVING_GEAR')] = -0.2828
        self.tracking_coeffs[('Paris','BERRIES')] = -0.1462
        self.tracking_coeffs[('Paris','DIP')] = 0.09721
        self.tracking_coeffs[('Paris','UKULELE')] = 0
        self.tracking_coeffs[('Paris','BAGUETTE')] = -0.3306
        
        
        self.tracking_coeffs[('Caesar','PICNIC_BASKET')] = 0.0719
        self.tracking_coeffs[('Caesar','BAGUETTE')] = 0.249
        self.tracking_coeffs[('Caesar','PINA_COLADAS')] = -0.32009
        self.tracking_coeffs[('Caesar','COCONUTS')] = 0.08314
        self.tracking_coeffs[('Caesar','DIP')] = 0
        self.tracking_coeffs[('Caesar','UKULELE')] = 0.1908
        self.tracking_coeffs[('Caesar','BANANAS')] = 0.32913

        
        self.tracking_coeffs[('Camilla','PICNIC_BASKET')] = -0.123
        self.tracking_coeffs[('Camilla','BERRIES')] = 0
        self.tracking_coeffs[('Camilla','BAGUETTE')] = 0
        self.tracking_coeffs[('Camilla','DIVING_GEAR')] = 0
        self.tracking_coeffs[('Camilla','DIP')] = -0.3012
        self.tracking_coeffs[('Camilla','UKULELE')] = -0.1
        self.tracking_coeffs[('Camilla','BANANAS')] =  -0.3052
        
        self.tracking_coeffs[('Charlie','PINA_COLADAS')] = -0.104
        self.tracking_coeffs[('Charlie','DIVING_GEAR')] = 0
        self.tracking_coeffs[('Charlie','BANANAS')] =  0.206
        self.tracking_coeffs[('Charlie','COCONUTS')] = -0.1061
        
        self.tracking_coeffs[('Pablo','PINA_COLADAS')] = -0.2590
        self.tracking_coeffs[('Pablo','BANANAS')] = -0.3953
        self.tracking_coeffs[('Pablo','PICNIC_BASKET')] = 0.3995
        self.tracking_coeffs[('Pablo','COCONUTS')] = 0.3162
          
        self.tracking_coeffs[('Penelope','BANANAS')] = -0.06591
        self.tracking_coeffs[('Penelope','BERRIES')] = -0.0567
        self.tracking_coeffs[('Penelope','PICNIC_BASKET')] = 0.38879

        self.tracking_coeffs[('Gary','BANANAS')] = 0.2930
        self.tracking_coeffs[('Gary','BERRIES')] = -0.3096

        self.tracking_coeffs[('Peter','PINA_COLADAS')] = -0.3952
        self.tracking_coeffs[('Peter','COCONUTS')] = -0.6835
        
        
        for trader in self.traders_tracked:
            for symbol in self.logger_monitored_symbols:
                self.tracking_coeffs[(trader,symbol)] = 0
        self.tracking_coeffs[('Olivia','UKULELE')] = 100
        self.tracking_coeffs[('Olivia','BERRIES')] = 1000
        self.tracking_coeffs[('Olivia','BANANAS')] = 5


class Logger:
        # This class logs the trading state of assets listed on the [monitored_symbol]
        # at each iteration. It writes the JSON representation of these states
        # by writing them to stdout.
    def __init__(self, config : Configuration()):
        self.monitored_symbols = config.logger_monitored_symbols

    def run(self, state: TradingState, file = None) -> Dict[str, List[Order]]:
        if len(self.monitored_symbols) == 0:
            return {}
        res = {}
        res["market_trades"] = {}
        res["own_trades"] = {}
        res["position"] = {}
        res["order_depths"] = {}
        res["observations"] = {}
        res["listings"] = {}
        res["timestamp"] = state.timestamp
        for symbol in self.monitored_symbols:
            if symbol in state.market_trades:
                res["market_trades"][symbol] = state.market_trades[symbol]
            if symbol in state.own_trades:
                res["own_trades"][symbol] = state.own_trades[symbol]
            if symbol in state.position:
                res["position"][symbol] = state.position[symbol]
            if symbol in state.order_depths:
                res["order_depths"][symbol] = state.order_depths[symbol]
            if symbol in state.listings:
                res["listings"][symbol] = state.listings[symbol]
            if symbol in state.observations:
                res["observations"][symbol] = state.observations[symbol]

        if file:
            file.write(str(state.timestamp) + ' ')
            file.write(json.dumps(res, default=lambda o: o.__dict__, sort_keys=True))
            file.write('\n')
        else:
            print(json.dumps(res, default=lambda o: o.__dict__, sort_keys=True))
        return {}

class HypotheticalPos:
    def __init__(self, init_pos :int):
        self.positive_pos = init_pos
        self.negative_pos = init_pos
        self.actual_pos = init_pos
        
class BaseMM:
    def __init__(self, spread, position_limit : int, trade_lot : int, symbol, survival_curve):
        self.pos_limit = position_limit
        self.symbol = symbol        
        self.alpha_plus = spread[1]
        self.alpha_min = spread[0]
        self.skew = 0
        self.trade_lot = trade_lot
        self.quoting_strat = self.greedy_quote
        self.hit_prob = survival_curve
        self.mid_price = 0

    def total_reward(self, n):
        if n >0:
            return -(self.alpha_plus -self.skew)*(1.0*n/self.pos_limit)**2
        else:
            return -(self.alpha_min -self.skew)*(1.0*n/self.pos_limit)**2

    def run(self, state:TradingState) -> Dict[str, List[Order]]:
        # print(state.order_depths)
        self.ingest_data(state)        
        
        if self.symbol not in state.order_depths:
            return {}

        # print('aa')

        # print('buy', state.order_depths[self.symbol].buy_orders)
        # print('sell', state.order_depths[self.symbol].sell_orders)
        curr_pos = state.position[self.symbol] if self.symbol in state.position else 0
        order = {}
        order[self.symbol] = []
        
        pos = HypotheticalPos(curr_pos)
        
        order_book = state.order_depths[self.symbol]
        old_order_book = deepcopy(order_book)
        buy_stack = list(order_book.buy_orders.items())
        buy_stack = sorted(buy_stack)
        sell_stack = list( order_book.sell_orders.items())
        sell_stack = sorted(sell_stack, reverse=True)
        
        while len(buy_stack) > 0: 
            best_buy = buy_stack[-1]
            taken = 0
            total_reward_diff = (self.total_reward(pos.actual_pos-1 ) - self.total_reward(pos.actual_pos))
            profit =  best_buy[0] - self.mid_price  
            while taken < best_buy[1] and pos.negative_pos > -self.pos_limit and total_reward_diff + profit > 0:
                taken += 1
                pos.actual_pos -= 1
                pos.negative_pos -= 1
                if pos.negative_pos > -self.pos_limit:
                    total_reward_diff =  (self.total_reward(pos.actual_pos-1) - self.total_reward(pos.actual_pos))
            order_book.buy_orders[best_buy[0]] -= taken
            if taken > 0:
                order[self.symbol].append(Order(self.symbol, best_buy[0], -taken))
            if pos.negative_pos == -self.pos_limit or taken < best_buy[1]:
                break
            elif taken == best_buy[1]:
                    buy_stack.pop()
                    del order_book.buy_orders[best_buy[0]] 
                
        while len(sell_stack) > 0: 
            best_sell = sell_stack[-1]
            taken = 0
            total_reward_diff = (self.total_reward(pos.actual_pos+1) - self.total_reward(pos.actual_pos))
            profit =  self.mid_price - best_sell[0]
            while taken < abs(best_sell[1]) and pos.positive_pos < self.pos_limit and total_reward_diff + profit > 0:
                taken += 1
                pos.actual_pos += 1
                pos.positive_pos += 1
                if pos.positive_pos < self.pos_limit:
                    total_reward_diff =  (self.total_reward(pos.actual_pos+1) - self.total_reward(pos.actual_pos))
            if taken >0:
                order[self.symbol].append(Order(self.symbol, best_sell[0], taken))
            order_book.sell_orders[best_sell[0]] += taken
            if pos.positive_pos == self.pos_limit or taken < abs(best_sell[1]):
                break
            elif taken == abs(best_sell[1]):
                sell_stack.pop()
                del order_book.sell_orders[best_sell[0]] 
        
        order[self.symbol] += self.quoting_strat(state, pos)
        state.order_depths[self.symbol] = old_order_book
        # print(self.quoting_strat(state, pos), state.order_depths[self.symbol].buy_orders, state.order_depths[self.symbol].sell_orders)
        return order
        
    def ingest_data(self, state:TradingState):
        pass
        
    def greedy_quote(self, state, pos):
        res = []        
        buy_orders = state.order_depths[self.symbol].buy_orders if self.symbol in state.order_depths else {}
        buy_orders = sorted(list(buy_orders.items()), reverse = True)        
        if len(buy_orders) > 0:
            cand_price = []
            cand_profit = []
            cand_obstacle = [0]
            prev_price = -1
            for order in buy_orders:
                cand_price.append(order[0]+1)
                cand_profit.append(self.mid_price-cand_price[-1])
                cand_obstacle.append(cand_obstacle[-1] + order[1])
            cand_price.append(buy_orders[-1][0])
            cand_profit.append(self.mid_price-cand_price[-1])
            
            alloc = self.solve_optimal_quote( cand_profit, cand_obstacle, pos.positive_pos, pos.actual_pos, True)
            for i, size in enumerate(alloc):
                if size >0:
                    res.append(Order(self.symbol, cand_price[i], size))
        
        sell_orders = state.order_depths[self.symbol].sell_orders if self.symbol in state.order_depths else {}
        sell_orders = sorted(list(sell_orders.items()))        
        if len(sell_orders) > 0:
            cand_price = []
            cand_profit = []
            cand_obstacle = [0]
            for order in sell_orders:
                cand_price.append(order[0]-1)
                cand_profit.append(cand_price[-1]-self.mid_price)
                cand_obstacle.append(cand_obstacle[-1] - order[1])
            cand_price.append(sell_orders[-1][0])
            cand_profit.append(self.mid_price-cand_price[-1])
            alloc = self.solve_optimal_quote( cand_profit, cand_obstacle, pos.negative_pos, pos.actual_pos, False)
            for i, size in enumerate(alloc):
                if size >0:
                    res.append(Order(self.symbol, cand_price[i], -size))
        return res        
        
    def no_quote(self, state, pos):
        return []
        
    def solve_optimal_quote(self, cand_profit, cand_obstacle, order_pos, actual_pos, is_bid):
        cand_alloc = [0]*len(cand_profit)
        
        hit_fun = lambda x: self.hit_prob[x] if x < len(self.hit_prob) else 0.0
        hit_prob = list(map(hit_fun, cand_obstacle))
        placed = 0
        while (placed < self.trade_lot) and (not((is_bid and order_pos == self.pos_limit) or (not is_bid and order_pos == -self.pos_limit))):
            placed += 1
            reward_diff = 0        
            if is_bid:
                reward_diff = self.total_reward(actual_pos + 1) - self.total_reward(actual_pos)
            else:
                reward_diff = self.total_reward(actual_pos - 1) - self.total_reward(actual_pos)
            obj_fun = list(map(lambda x: (hit_prob[x]*(cand_profit[x] + reward_diff),x),  range(len(hit_prob))))
            
            best_pos = max(obj_fun)
            if best_pos[0] <= 0:
                break
            cand_alloc[best_pos[1]] += 1
            if is_bid:
                order_pos +=1
                actual_pos +=1
            else:
                order_pos -=1
                actual_pos -= 1
            for i in range(best_pos[1], len(cand_alloc)):
                cand_obstacle[i] += 1
                hit_prob[i] = hit_fun(cand_obstacle[i])
        return cand_alloc

class MeasurementMM(BaseMM):
    def __init__(self, pos_limit, symbol, measured_spread):
        super().__init__((0.13,0), pos_limit,0, symbol,[])
        self.measured_spread = measured_spread
        self.quoting_strat = self.measurement_quote

    def measurement_quote(self, state, curr_pos):
        if self.symbol not in state.order_depths:
            return []
        res = []
        buy_price = round(self.mid_price) -self.measured_spread
        sell_price =round(self.mid_price) +self.measured_spread
        buy_vol = self.pos_limit - curr_pos.positive_pos
        sell_vol = self.pos_limit + curr_pos.negative_pos
        if buy_vol > 0:
            res.append(Order(self.symbol, buy_price, buy_vol))
        if sell_vol > 0:
            res.append(Order(self.symbol, sell_price, -sell_vol))
        return res

    def ingest_data(self, state):
        self.mid_price = 10000

class PearlMM(BaseMM):
    def __init__(self, config = Configuration()):
        super().__init__(config.pearl_mm_spread, 20,20, "PEARLS", config.pearl_mm_survival_curve)

    def ingest_data(self, state: TradingState):
        self.mid_price = 10000



class BananaMM(BaseMM):
    def __init__(self, config = Configuration(), trader_tracker = None):
        super().__init__(config.banana_mm_spread, 20,20, "BANANAS", config.banana_mm_survival_curve)
        self.mid_price = 4940
        self.trader_tracker = trader_tracker
        self.config = config
        
    def ingest_data(self, state: TradingState):
        try:
            order_book = state.order_depths["BANANAS"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            midprice = (outer_bid[0] + outer_ask[0])/2
        
            self.mid_price = midprice
            self.t = state.timestamp
              
        except:
            pass

    def total_reward(self, k):
        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'BANANAS')][1]*self.config.tracking_coeffs[(trader,'BANANAS')], self.pos_limit),-self.pos_limit)

        if k >0:
            res =  -(self.alpha_plus -self.skew)*(1.0*(k-rel_pos)/self.pos_limit)**2
        else:
            res = -(self.alpha_min -self.skew)*(1.0*(k-rel_pos)/self.pos_limit)**2

        # if self.trader_tracker:
        #     for trader in self.config.traders_tracked:
        #         res += np.exp(-self.config.forget_factor*(self.t-self.trader_tracker.tracking_dict[(trader,'BANANAS')][0]))*self.trader_tracker.tracking_dict[(trader,'BANANAS')][1]*self.config.tracking_coeffs[(trader,'BANANAS')]*k/10

        return res


class PinaCocoKF:
    def __init__(self,config = Configuration()):
        self.rate_mean = np.nan
        self.rate_var = 0
        self.Q1 = np.exp(-3*config.pinacoco_kf_Q1)
        self.Q2 = 20*config.pinacoco_kf_Q2+1.0
        self.curr_spread = 0
        self.curr_timestamp = -1

        
    def update(self, state):
        if True:
            if self.curr_timestamp >= state.timestamp:
                return
            
            self.curr_timestamp = state.timestamp;
            order_book = state.order_depths["PINA_COLADAS"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            pina_price = (outer_bid[0] + outer_ask[0])/2
            order_book = state.order_depths["COCONUTS"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            coco_price = (outer_bid[0] + outer_ask[0])/2

            self.pina_midprice = pina_price
            self.coco_midprice = coco_price            
            pina_price = pina_price/5000
            coco_price = coco_price/5000
            if np.isnan(self.rate_mean) and coco_price >0:
                self.rate_mean = 1.8755
            else:
                param_var = 0.99*self.rate_var*0.99 + self.Q1
                self.rate_mean = self.rate_mean*0.99 + 0.01*1.875
                err = pina_price - coco_price*self.rate_mean
                err_var = coco_price*param_var*coco_price + self.Q2
                kalman_gain = param_var*coco_price/err_var
                self.rate_mean = self.rate_mean + kalman_gain*err
                self.rate_var = (1-kalman_gain*coco_price)*param_var
                self.curr_spread = pina_price - coco_price*self.rate_mean
                self.curr_spread = self.curr_spread*5000
            
class PinaMM(BaseMM):
    def __init__(self, pina_coco_kf : PinaCocoKF, config = Configuration(), trader_tracker = None):
        super().__init__((0,0), 300,20, "PINA_COLADAS", config.pina_mm_survival_curve)
        self.mid_price = 15000
        self.config = config
        self.kf = pina_coco_kf
        self.vol_target = 0
        self.trader_tracker = trader_tracker

    def ingest_data(self, state: TradingState):
        try:
            
            order_book = state.order_depths["PINA_COLADAS"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            pina_price = (outer_bid[0] + outer_ask[0])/2
            self.kf.update(state)
            coco_pos = state.position["COCONUTS"] if "COCONUTS" in state.position else 0
            self.vol_target =  -coco_pos/self.kf.rate_mean
            self.price_spread = self.kf.curr_spread
            self.mid_price = pina_price
            self.t = state.timestamp


        except:
            pass
            
    def total_reward(self, k):
        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'PINA_COLADAS')][1]*self.config.tracking_coeffs[(trader,'PINA_COLADAS')], self.pos_limit),-self.pos_limit)

        res = -self.config.pina_mm_spread_beta*self.price_spread*k
        res -= self.config.pina_mm_vol_diff*(k-self.vol_target)**2
        res -= self.config.pina_mm_spread*((k-rel_pos))**2

        # if self.trader_tracker:
        #     for trader in self.config.traders_tracked:
        #         res += np.exp(-self.config.forget_factor*(self.t-self.trader_tracker.tracking_dict[(trader,'PINA_COLADAS')][0]))*self.trader_tracker.tracking_dict[(trader,'PINA_COLADAS')][1]*self.config.tracking_coeffs[(trader,'PINA_COLADAS')]*k/10

        return res

class CoconutMM(BaseMM):
    def __init__(self, pina_coco_kf : PinaCocoKF, config = Configuration(), trader_tracker = None):
        super().__init__((0,0), 600, 50, "COCONUTS", config.coco_mm_survival_curve)
        self.mid_price = 8000
        self.config = config
        self.trader_tracker = trader_tracker
        self.kf = pina_coco_kf
        self.vol_target = 0

    def ingest_data(self, state: TradingState):
        try:
            self.kf.update(state)
            
            order_book = state.order_depths["COCONUTS"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            coco_price = (outer_bid[0] + outer_ask[0])/2
            pina_pos = state.position["PINA_COLADAS"] if "PINA_COLADAS" in state.position else 0
            self.vol_target =  -self.kf.rate_mean*pina_pos
            self.price_spread = self.kf.curr_spread
            
            self.mid_price = coco_price
            self.t = state.timestamp
                  
        except:
            pass
        
    def total_reward(self, k):
        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'COCONUTS')][1]*self.config.tracking_coeffs[(trader,'COCONUTS')], self.pos_limit),-self.pos_limit)

        res = self.config.coco_mm_spread_beta*self.price_spread*k
        res -= self.config.coco_mm_vol_diff*(k-self.vol_target)**2
        res -= self.config.coco_mm_spread*((k-rel_pos))**2

        # if self.trader_tracker:
        #     for trader in self.config.traders_tracked:
        #         res += np.exp(-self.config.forget_factor*(self.t-self.trader_tracker.tracking_dict[(trader,'COCONUTS')][0]))*self.trader_tracker.tracking_dict[(trader,'COCONUTS')][1]*self.config.tracking_coeffs[(trader,'COCONUTS')]*k/10

        return res             
        
class GearMM(BaseMM):
    def __init__(self, config = Configuration(), trader_tracker = None):
        super().__init__((0,0), 50, 8, "DIVING_GEAR", config.gear_mm_survival_curve)
        self.mid_price = 8000
        
        self.obs_long = 0
        self.obs_long_alpha = 1.0/(1+100*config.gear_mm_long)
        self.obs_short = 0
        self.obs_short_alpha = 1.0/(1+10*config.gear_mm_short)
        self.price_long = 0
        self.price_long_alpha = 1.0/(1+100*config.gear_mm_long)
        self.price_short = 0
        self.price_short_alpha = 1.0/(1+10*config.gear_mm_short)
        self.config = config
        self.trader_tracker = trader_tracker

    def ingest_data(self, state: TradingState):        
        try:
            order_book = state.order_depths["DIVING_GEAR"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            midprice = (outer_bid[0] + outer_ask[0])/2
            self.t = state.timestamp
            
            self.mid_price = midprice
            obs = state.observations["DOLPHIN_SIGHTINGS"]
            if self.obs_short == 0:
                self.obs_short = obs
                self.obs_long = obs
                self.price_long = midprice
                self.price_short = midprice
            else:
                self.obs_short *= (1-self.obs_short_alpha)
                self.obs_short += self.obs_short_alpha*obs
                self.obs_long *= (1-self.obs_long_alpha)
                self.obs_long += self.obs_long_alpha*obs
                self.price_short *= (1-self.price_short_alpha)
                self.price_short += self.price_short_alpha*midprice
                self.price_long *= (1-self.price_long_alpha)
                self.price_long += self.price_long_alpha*midprice
        except:
            pass

    def total_reward(self, k):

        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'DIVING_GEAR')][1]*self.config.tracking_coeffs[(trader,'DIVING_GEAR')], self.pos_limit),-self.pos_limit)

        res = -4*self.config.gear_mm_spread*( (k-rel_pos) )**2
        res += 20*self.config.gear_mm_obs_bias*(self.obs_short - self.obs_long)*k
        res += self.config.gear_mm_price_bias*(self.price_short - self.price_long)*k

        # if self.trader_tracker:
        #     for trader in self.config.traders_tracked:
        #         res += np.exp(-self.config.forget_factor*(self.t-self.trader_tracker.tracking_dict[(trader,'DIVING_GEAR')][0]))*self.trader_tracker.tracking_dict[(trader,'DIVING_GEAR')][1]*self.config.tracking_coeffs[(trader,'DIVING_GEAR')]*k/10

        return res
     
    
class BerryMM(BaseMM):
    def __init__(self, config = Configuration(), trader_tracker = None):
        super().__init__((0,0), 250,50, "BERRIES", config.berry_mm_survival_curve)
        self.mid_price = 3900
        self.config = config
        self.trader_tracker = trader_tracker
        self.price_spread = 0
        self.vol_target = 0
        # self.t = 0

    def ingest_data(self, state: TradingState):
        try:
        
            order_book = state.order_depths["BERRIES"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            berry_midprice = (outer_bid[0] + outer_ask[0])/2
            self.t = state.timestamp
            
            berry_pos = state.position["BERRIES"] if "BERRIES" in state.position else 0
        
            self.mid_price = berry_midprice
        except:
            pass
        
            
    def total_reward(self, k):

        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'BERRIES')][1]*self.config.tracking_coeffs[(trader,'BERRIES')], self.pos_limit),-self.pos_limit)
        print(rel_pos)
        res = -0.5*abs((k-rel_pos)/10.0)**3
        #(real best one)
        buy_line = -(self.t-5.2e5)*np.exp(-((self.t-5.2e5)/60000)**2)*(k/10.0)*100

        res += buy_line

        # if self.trader_tracker:
        #     for trader in self.config.traders_tracked:
        #         res += np.exp(-self.config.forget_factor*(self.t-self.trader_tracker.tracking_dict[(trader,'BERRIES')][0]))*self.trader_tracker.tracking_dict[(trader,'BERRIES')][1]*self.config.tracking_coeffs[(trader,'BERRIES')]*k/10

        # res += np.exp(-self.config.forget_factor*(self.t-self.trader_tracker.tracking_dict[('Camilla','BERRIES')][0]))*self.trader_tracker.tracking_dict[('Camilla','BERRIES')][1]*self.config.tracking_coeffs[('Camilla','BERRIES')]*k/10

        # res += 


        return res

class DipMM(BaseMM):
    def __init__(self, config = Configuration(),trader_tracker = None):
        super().__init__((0,0), 300,15, "DIP", config.dip_mm_survival_curve)
        self.mid_price = 7000
        self.config = config
        self.price_spread = 0
        self.vol_target = 0
        self.basket_pos = 0
        self.config = config
        self.trader_tracker = trader_tracker

    def ingest_data(self, state: TradingState):
        
        if 'DIP' in state.order_depths:
            order_book = state.order_depths["DIP"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            midprice = (outer_bid[0] + outer_ask[0])/2
            self.mid_price = midprice
            self.basket_pos = state.position["PICNIC_BASKET"] if "PICNIC_BASKET" in state.position else 0

            
    def total_reward(self, k):

        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'DIP')][1]*self.config.tracking_coeffs[(trader,'DIP')], self.pos_limit),-self.pos_limit)

        # res = -self.config.pina_mm_spread_beta*self.price_spread*k/10
        # res -= self.config.pina_mm_vol_diff*((k-self.vol_target)/10.0)**2
        res = -self.config.dip_mm_spread*abs((k+4*self.basket_pos-rel_pos)/10.0)**2

        return res

class BaguetteMM(BaseMM):
    def __init__(self, config = Configuration(),trader_tracker = None):
        super().__init__((0,0), 150,10, "BAGUETTE", config.baguette_mm_survival_curve)
        self.mid_price = 12000
        self.config = config
        self.price_spread = 0
        self.vol_target = 0
        self.basket_pos = 0
        self.baguette_pos= 0
        self.config = config
        self.trader_tracker = trader_tracker
        # self.t = 0

    def ingest_data(self, state: TradingState):
        
            
        if 'BAGUETTE' in state.order_depths:
            order_book = state.order_depths["BAGUETTE"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            midprice = (outer_bid[0] + outer_ask[0])/2
            self.mid_price = midprice
            self.basket_pos = state.position["PICNIC_BASKET"] if "PICNIC_BASKET" in state.position else 0
            
    def total_reward(self, k):
        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'BAGUETTE')][1]*self.config.tracking_coeffs[(trader,'BAGUETTE')], self.pos_limit),-self.pos_limit)

        res = -self.config.baguette_mm_spread*abs((k+2*self.basket_pos-rel_pos)/10.0)**2

        return res
        
class UkuleleMM(BaseMM):
    def __init__(self, config = Configuration(),trader_tracker = None):
        super().__init__((0,0), 70,3, "UKULELE", np.ones((50,1)))
        self.mid_price = 20800
        self.config = config
        self.price_spread = 0
        self.vol_target = 0
        self.basket_pos = 0
        self.ukulele_pos = 0
        self.config = config
        self.trader_tracker = trader_tracker
        # self.t = 0

    def ingest_data(self, state: TradingState):
        
            
        if 'UKULELE' in state.order_depths:
            # print('uu')
            order_book = state.order_depths["UKULELE"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            midprice = (outer_bid[0] + outer_ask[0])/2
            self.mid_price = midprice
            self.basket_pos = state.position["PICNIC_BASKET"] if "PICNIC_BASKET" in state.position else 0
                   
        
            
    def total_reward(self, k):
        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'UKULELE')][1]*self.config.tracking_coeffs[(trader,'UKULELE')], self.pos_limit),-self.pos_limit)
        # res = -self.config.pina_mm_spread_beta*self.price_spread*k/10
        # res -= self.config.pina_mm_vol_diff*((k-self.vol_target)/10.0)**2

        res = -self.config.ukulele_mm_spread*abs((k+self.basket_pos-rel_pos)/10.0)**2
        return res



class BasketMM(BaseMM):
    def __init__(self, config = Configuration(),trader_tracker = None):
        super().__init__((0,0), 70,15, "PICNIC_BASKET", config.basket_mm_survival_curve)
        self.mid_price = 73365
        self.config = config
        self.price_spread = 0
        self.vol_target = 0
        self.spread = 0
        self.config = config
        self.empty_basket_price = self.config.init_basket_price
        self.alpha = np.exp(-self.config.basket_mm_ewma_length)
        self.trader_tracker = trader_tracker


    def ingest_data(self, state: TradingState):          
        if 'PICNIC_BASKET' in state.order_depths:
            order_book = state.order_depths["PICNIC_BASKET"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            midprice = (outer_bid[0] + outer_ask[0])/2
            self.mid_price = midprice
            
            order_book = state.order_depths["DIP"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            dip_price = (outer_bid[0] + outer_ask[0])/2
            
            order_book = state.order_depths["BAGUETTE"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            baguette_price = (outer_bid[0] + outer_ask[0])/2
            
            order_book = state.order_depths["UKULELE"]
            outer_bid = min(order_book.buy_orders.items())
            outer_ask = max(order_book.sell_orders.items())
            ukulele_price = (outer_bid[0] + outer_ask[0])/2
            
            raw_spread = midprice - (baguette_price*2 + dip_price*4 + ukulele_price)
            if self.empty_basket_price is None :           
                self.spread = 0
                self.empty_basket_price = raw_spread
            else:            
                self.spread = raw_spread - self.empty_basket_price
                self.empty_basket_price = (1-self.alpha)*self.empty_basket_price +  self.alpha*raw_spread

            
    def total_reward(self, k):

        rel_pos = 0

        if self.trader_tracker:
            for trader in self.config.traders_tracked:
                rel_pos += max(min(self.trader_tracker.tracking_dict[(trader,'PICNIC_BASKET')][1]*self.config.tracking_coeffs[(trader,'PICNIC_BASKET')], self.pos_limit),-self.pos_limit)

        # res = -self.config.pina_mm_spread_beta*self.price_spread*k/10
        # res -= self.config.pina_mm_vol_diff*((k-self.vol_target)/10.0)**2
        res = -self.config.basket_mm_spread*abs((k-rel_pos)/10.0)**2
        res -= self.config.basket_mm_bias*k*self.spread

        return res

class TraderTracker:

    def __init__(self, config = Configuration()):
        self.config = config
        self.traders_tracked = self.config.traders_tracked
        self.logger_monitored_symbols = self.config.logger_monitored_symbols

        self.tracking_dict = {}

        for trader in self.traders_tracked:
            for symbol in self.logger_monitored_symbols:
                self.tracking_dict[(trader,symbol)] = [0,0]

        # print(self.tracking_dict)
        # print(self.logger_monitored_symbols)

    def update(self, state : TradingState):

        market_trades = state.market_trades
        timestamp = state.timestamp

        for symbol,trade_list in market_trades.items():
            if symbol in self.logger_monitored_symbols:
                for trade in trade_list:
                    if trade.buyer in self.traders_tracked:
                        # if self.tracking_dict[(trade.buyer,symbol)][0] < timestamp:
                        self.tracking_dict[(trade.buyer,symbol)][0] = timestamp
                        self.tracking_dict[(trade.buyer,symbol)][1] += 1

                    if trade.seller in self.traders_tracked:
                        # if self.tracking_dict[(trade.seller,symbol)][0] < timestamp:
                        self.tracking_dict[(trade.seller,symbol)][0] = timestamp
                        self.tracking_dict[(trade.seller,symbol)][1] -= 1

        # print(state.market_trades)
        # print(self.tracking_dict)
     
class Trader:
    def __init__(self, config = Configuration()):
        self.traders = []
        self.trader_tracker = TraderTracker(config)
        self.kf = PinaCocoKF(config)
        #self.traders.append(MeasurementMM(20, "PEARLS",2))
        self.traders.append(BananaMM(config = config, trader_tracker = self.trader_tracker))
        self.traders.append(PearlMM(config))

        self.traders.append(PinaMM(self.kf,config,trader_tracker = self.trader_tracker))
        self.traders.append(CoconutMM(self.kf,config,trader_tracker = self.trader_tracker))
        self.traders.append(GearMM(config,trader_tracker = self.trader_tracker))
        self.traders.append(BerryMM(config,trader_tracker = self.trader_tracker))

        self.traders.append(DipMM(config,trader_tracker = self.trader_tracker))
        self.traders.append(UkuleleMM(config,trader_tracker = self.trader_tracker))
        self.traders.append(BaguetteMM(config,trader_tracker = self.trader_tracker))
        self.traders.append(BasketMM(config,trader_tracker = self.trader_tracker))

        # self.traders.append(Logger(config))

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # We pass the state through a list of trader.
        # Each of them will generate an order dict.

        orders = []
        for trader in self.traders:
            self.trader_tracker.update(state)
            orders.append(trader.run(state))

        # Combine the order dict from each traders to a final order dict
        #  We might add internalization later on if we need to do cross-asset hedging
        final_order = {}
        for order in orders:
            for product in order:
                if product in final_order:
                    final_order[product] += order[product]
                else:
                    final_order[product] = order[product]
        return final_order
