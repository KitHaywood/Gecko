from pycoingecko import CoinGeckoAPI
import datetime as dt
import pandas as pd
import time
import requests
import json
import tqdm
import os
from utils import date_range_lister
# TODO - Go back and use market_chart_range to get 
 
class Gecko:
    
    def __init__(self) -> None:
        self.cg = CoinGeckoAPI()

    def get_all_coins(self):
        return self.cg.get_coins_list()
    
    def get_market_data(self,symbol,currency,days):
        base_url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency={currency}&days={days}'
        response = requests.get(base_url)
        res = response.json()
        data = res['prices']
        data = [[dt.datetime.fromtimestamp(x[0]/1000),x[1]] for x in data]
        df = pd.DataFrame(data)
        if days=='max':
            df[0] = df[0].apply(lambda x: dt.datetime.strptime(x.strftime("%d-%m-%Y"),"%d-%m-%Y"))
            df.index = df[0]
            df = df[1]
        else:
            df[0] = df[0].apply(lambda x: dt.datetime.strptime(x.strftime("%d-%m-%YT%H:%M:%S"),"%d-%m-%YT%H:%M:%S"))
            df.index = df[0]
            df = df[1]
        return df

    def write_original_to_json(self,crypto,currency,days):
        with open(os.path.join(os.getcwd(),f'{crypto}.json'),'w') as f:
            data = self.get_market_data(crypto,currency,days) # CAREFUL - need an overwrite function
            json.dump(data.to_json(orient='split',date_format='iso'),f)
    
    def get_dates(self,crypto):
        with open(os.path.join(os.getcwd(),f'{crypto}.json'),'r') as f:
            data = json.load(f)
            data = json.loads(data)
            res = pd.DataFrame.from_dict(data,orient='columns')
            res.index = res['index']
            return res.index.max(), res.index.min()
    
    def get_data_on_interval(self,symbol,currency,from_date,to_date):
        """
        takes symbol, target return currency, from and to dates and returns

        df of prices on the interval
        """
        base_url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range?vs_currency={currency}&from={from_date}&to={to_date}"
        response = requests.get(base_url)
        res = response.json()
        data = res['prices']
        data = [[dt.datetime.fromtimestamp(x[0]/1000),x[1]] for x in data]
        df = pd.DataFrame(data)
        return df
    
    def write_to_json(self,crypto,currency):
        # NOT WORKING YET - JUST SCRIBBLING
        dates = date_range_lister()
        res = pd.DataFrame()
        for i in tqdm.tqdm(range(len(dates))):
            if i==0:
                pass
            else:
                data = self.get_data_on_interval(crypto,currency,dates[i],dates[i-1])
                res = pd.concat([res,data])    
            time.sleep(0.75)
        
        return res


if __name__=="__main__":

    cryptos = ['bitcoin','ethereum','tether','cardarno','usd-coin']



    # ETH = Gecko().get_market_data('ethereum','usd','max')
    # print(ETH)

    gck = Gecko()
    # gck.write_original_to_json('ethereum','usd','1')
    # print(gck.get_dates('ethereum'))
    print(gck.write_to_json('ethereum','usd'))

    # gck.write_original_to_json('ethereum','usd','max')
    # print(gck.get_dates('ethereum'))

    # coin_list = Gecko().get_all_coins()
    # print(coin_list)
    # IDs = []
    # for coin in coin_list:
    #     IDs.append(coin['id'])
    # print(IDs)
    # BTC = Gecko().get_100d_history('bitcoin')
    # print(BTC)
    
    # print(BTC.keys())
    # BTC = Gecko().get_history('bitcoin')
    # print(BTC)

    # def get_100d_history_test(self,symbol,date):
    #     dates = pd.date_range(end = dt.datetime.today(), periods = 100).to_pydatetime().tolist()
    #     dates = [x.strftime('%d-%m-%Y') for x in dates]
    #     res = pd.DataFrame()
    #     # for date in dates:
    #     all_data =  self.cg.get_coin_history_by_id(symbol,date)
    #     market_data = all_data['market_data']['current_price']
    #     print(list(market_data.keys()))
    #     # res.loc[date] = 
    
    # def get_history(self,symbol,period):
    #     """
    #     takes symbol as lower full str name of crypto and period=days into past
    #     """
    #     dates = pd.date_range(end = dt.datetime.today(), periods = period).to_pydatetime().tolist()
    #     dates = [x.strftime('%d-%m-%Y') for x in dates]
    #     columns=['aed', 'ars', 'aud', 'bch', 'bdt', 'bhd', 'bmd', 'bnb', 'brl', 'btc', 'cad', 'chf', 'clp', 'cny', 'czk', 'dkk', 'dot', 'eos', 'eth', 'eur', 'gbp', 'hkd', 'huf', 'idr', 'ils', 'inr', 'jpy', 'krw', 'kwd', 'lkr', 'ltc', 'mmk', 'mxn', 'myr', 'ngn', 'nok', 'nzd', 'php', 'pkr', 'pln', 'rub', 'sar', 'sek', 'sgd', 'thb', 'try', 'twd', 'uah', 'usd', 'vef', 'vnd', 'xag', 'xau', 'xdr', 'xlm', 'xrp', 'yfi', 'zar', 'bits', 'link', 'sats']
    #     res_dict = {}
    #     for date in tqdm.tqdm(dates):
    #         all_data =  self.cg.get_coin_history_by_id(symbol,date)
    #         market_data = list([round(float(x),2) for x in all_data['market_data']['current_price'].values()])
    #         res_dict[date] = market_data
    #         time.sleep(0.75)
    #     res = pd.DataFrame.from_dict(res_dict,orient='index',columns=columns)

    #     return res