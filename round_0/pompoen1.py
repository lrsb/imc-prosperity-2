from datamodel import Trade, Listing, OrderDepth, UserId, TradingState, Order
from typing import List
# import string
# import pandas as pd
import jsonpickle

class Trader:
    # self.
    
    def run(self, state: TradingState):
        limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        # print('-----')
        # print(jsonpickle.dumps(state, indent=4))
        
        # print("traderData: " + state.traderData)
        # print(jsonpickle.dumps(observations, indent=4))

        if state.traderData == '' or state.traderData == None:
            trader_data = {}
            for product in state.order_depths:
                trader_data[product] = {'prices': [ ( max( state.order_depths[product].buy_orders ) + min( state.order_depths[product].sell_orders ) ) / 2 ]}
        else:
            trader_data = jsonpickle.loads(state.traderData)

        # print('trader_data')
        # print(trader_data)
        result = {}
        for product in state.order_depths:
            # print('-----------')
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []



            fair_price = self.calculate_fair_price(state, product, trader_data);  # Participant should calculate this value


            # print(state.position)

            # BUY
            asks = list(sorted(order_depth.sell_orders.items()))
            best_ask, best_ask_amount = asks[0]

            order_amount_buy = -best_ask_amount
            max_order_amount_buy = limits[product] - state.position.get(product, 0)
            amount_buy = order_amount_buy if abs(order_amount_buy) < abs(max_order_amount_buy) else max_order_amount_buy

            # SELL
            bids = list(sorted(order_depth.buy_orders.items(), reverse=True))
            best_bid, best_bid_amount = bids[0]

            order_amount_sell = -best_bid_amount
            max_order_amount_sell = -limits[product] - state.position.get(product, 0)
            amount_sell = order_amount_sell if abs(order_amount_sell) < abs(max_order_amount_sell) else max_order_amount_sell

            # ORDERS
            if len(order_depth.sell_orders) != 0:
                if ( int(best_ask) < fair_price or ( product == 'AMETHYSTS' and int(best_ask) < fair_price - 1 ) ) and amount_buy > 0:
                    orders.append(Order(product, best_ask, amount_buy)) # BUY

                    # See if the second best bid is also below the fair price, if so hit the ask
                    if len(asks) >= 2:
                        best_ask2, best_ask_amount2 = asks[1]

                        order_amount_buy2 = -best_ask_amount2
                        max_order_amount_buy2 = max_order_amount_buy - amount_buy
                        amount_buy2 = order_amount_buy2 if order_amount_buy2 <= max_order_amount_buy2 else max_order_amount_buy2

                        # Hit the second best ask if it's beyond the fair price
                        if int(best_ask2) < fair_price and amount_buy2 > 0:
                            orders.append(Order(product, best_ask2, amount_buy2)) # BUY

                        # If you have space left within your position limits, add a limit order to the order book
                        elif max_order_amount_buy2 > 0:
                            if int(best_bid) < fair_price - 2:
                                orders.append(Order(product, best_bid + 1, max_order_amount_buy2)) # BUY
                            elif int(best_bid) < fair_price - 1:
                                orders.append(Order(product, best_bid, max_order_amount_buy2)) # BUY
                            else:
                                orders.append(Order(product, round(fair_price) - 2, max_order_amount_buy2)) # BUY
                        

                # Put a bid below the fair price, match the best bid, or try to improve the best bid
                elif max_order_amount_buy > 0:
                    if int(best_bid) < fair_price - 2:
                        orders.append(Order(product, best_bid + 1, max_order_amount_buy)) # BUY
                    elif int(best_bid) < fair_price - 1:
                        orders.append(Order(product, best_bid, max_order_amount_buy)) # BUY
                    else:
                        orders.append(Order(product, round(fair_price) - 2, max_order_amount_buy)) # BUY
                        

            if len(order_depth.buy_orders) != 0:
                if ( int(best_bid) > fair_price or ( product == 'AMETHYSTS' and int(best_bid) > fair_price - 1 ) ) and amount_sell < 0:
                    orders.append(Order(product, best_bid, amount_sell)) # SELL

                    # See if the second best bid is also above the fair price, if so hit the bid
                    if len(bids) >= 2:
                        best_bid2, best_bid_amount2 = bids[1]

                        order_amount_sell2 = -best_bid_amount2
                        max_order_amount_sell2 = max_order_amount_sell - amount_sell
                        amount_sell2 = order_amount_sell2 if order_amount_sell2 <= max_order_amount_sell2 else max_order_amount_sell2

                        # Hit the second best bid if it's beyond the fair price
                        if int(best_bid2) > fair_price and amount_sell2 < 0:
                            orders.append(Order(product, best_bid2, amount_sell2)) # SELL

                        # If you have space left within your position limits, add a limit order to the order book
                        elif max_order_amount_sell2 < 0:
                            if int(best_ask) > fair_price + 2:
                                orders.append(Order(product, best_ask - 1, max_order_amount_sell2)) # SELL
                            elif int(best_ask) > fair_price + 1:
                                orders.append(Order(product, best_ask, max_order_amount_sell2)) # SELL
                            else:
                                orders.append(Order(product, round(fair_price) + 2, max_order_amount_sell2)) # SELL

                # Put an ask above the fair price, match the best ask, or try to improve the best ask
                elif max_order_amount_sell < 0:
                    if int(best_ask) > fair_price + 2:
                        orders.append(Order(product, best_ask - 1, max_order_amount_sell)) # SELL
                    elif int(best_ask) > fair_price + 1:
                        orders.append(Order(product, best_ask, max_order_amount_sell)) # SELL
                    else:
                        orders.append(Order(product, round(fair_price) + 2, max_order_amount_sell)) # SELL

        
            result[product] = orders


        conversions = 1
        # print(jsonpickle.dumps(result, indent=4))
        return result, conversions, jsonpickle.dumps(trader_data)


    def calculate_fair_price(self, state: TradingState, product, trader_data: dict):
        buy_orders = list( sorted( state.order_depths[product].buy_orders, reverse=True ) )
        sell_orders = list( sorted( state.order_depths[product].sell_orders ) )

        fair_price_this_run = ( buy_orders[0] + sell_orders[0] ) / 2
        prices = trader_data[product]['prices']
        prices.append( fair_price_this_run )

        ma_length = 10
        if product == 'AMETHYSTS':
            ma_length = 100

        if len(prices) > ma_length:
            prices = prices[-ma_length:]

        average_fair_price = sum(prices) / len(prices)

        trader_data[product]['prices'] = prices
        return average_fair_price


if __name__ == "__main__":
    # timestamp = 1000

    listings = {
        "STARFRUIT": Listing(
            symbol="STARFRUIT", 
            product="STARFRUIT", 
            denomination= "SEASHELLS"
        ),
        "AMETHYSTS": Listing(
            symbol="AMETHYSTS", 
            product="AMETHYSTS", 
            denomination= "SEASHELLS"
        ),
    }

    # order_depths = {
    #     "PRODUCT1": OrderDepth(
    #         buy_orders={10: 7, 9: 5},
    #         sell_orders={11: -4, 12: -8}
    #     ),
    #     "PRODUCT2": OrderDepth(
    #         buy_orders={142: 3, 141: 5},
    #         sell_orders={144: -5, 145: -8}
    #     ),	
    # }

    own_trades = {
        "STARFRUIT": [],
        "AMETHYSTS": []
    }

    market_trades = {
        "STARFRUIT": [
            Trade(
                symbol="STARFRUIT",
                price=11,
                quantity=4,
                buyer="",
                seller="",
                timestamp=900
            )
        ],
        "AMETHYSTS": []
    }

    position = {
        "STARFRUIT": -20,
        "AMETHYSTS": -20
    }

    observations = {}
    traderData = ""



    import re
    import pandas as pd

    df = pd.read_csv('price-data.csv', sep=';')

    all_orders = []

    import numpy as np
    for timestamp in df['timestamp'].unique():
        if timestamp >= 20000:
            break
        print('---', timestamp, '---')
        daily_df = df.loc[df['timestamp'] == timestamp, :]
        columns = daily_df.columns

        order_depths = {}
        for i, row in daily_df.iterrows():
            buy_orders = {}
            sell_orders = {}

            for col in columns:
                if 'bid_price_' in col:
                    if f'{row[col]}' == 'nan':
                        continue
                    buy_orders[row[col]] = row['bid_volume_' + re.findall('[0-9]+', col)[0]]
                elif 'ask_price_' in col:
                    if f'{row[col]}' == 'nan':
                        continue
                    sell_orders[row[col]] = -row['ask_volume_' + re.findall('[0-9]+', col)[0]]
            
            order_depths[row['product']] = OrderDepth(
                buy_orders=buy_orders,
                sell_orders=sell_orders
            )
        

        state = TradingState(
            traderData,
            timestamp,
            listings,
            order_depths,
            own_trades,
            market_trades,
            position,
            observations
        )


        trader = Trader()
        result, conversions, traderData = trader.run(state)

        all_orders.append(result)

    # print(all_orders)
        





"""
    
    "listings": {
        "PRODUCT1": {
            "py/object": "datamodel.Listing",
            "symbol": "PRODUCT1",
            "product": "PRODUCT1",
            "denomination": "SEASHELLS"
        },
        "PRODUCT2": {
            "py/object": "datamodel.Listing",
            "symbol": "PRODUCT2",
            "product": "PRODUCT2",
            "denomination": "SEASHELLS"
        }
    },
    "order_depths": {
        "STARFRUIT": {
            "py/object": "datamodel.OrderDepth",
            "buy_orders": {
                "4997": 2,
                "4996.0": 20.0
            },
            "sell_orders": {
                "5003": 20
            }
        },
        "AMETHYSTS": {
            "py/object": "datamodel.OrderDepth",
            "buy_orders": {
                "9995": 20
            },
            "sell_orders": {
                "10005": 20
            }
        }
    },
    "own_trades": {
        "PRODUCT1": [],
        "PRODUCT2": []
    },
    "market_trades": {
        "PRODUCT1": [
            {
                "py/object": "datamodel.Trade",
                "symbol": "PRODUCT1",
                "price": 11,
                "quantity": 4,
                "buyer": "",
                "seller": "",
                "timestamp": 900
            }
        ],
        "PRODUCT2": []
    },
    "position": {
        "PRODUCT1": 3,
        "PRODUCT2": -5
    },
    "observations": {}
}
    """