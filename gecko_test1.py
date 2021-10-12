try:
    from pycoingecko import CoinGeckoAPI
except ImportError:
    print('please install pycoingecko in the CLI')
import datetime as dt
try:
    import pandas as pd
except ImportError:
    print('please install pandas in the CLI'')
import time
import requests
import json
import tqdm
import os
from utils import date_range_lister

# TODO - Write the dox uuuhhhhhh
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
        """
        takes crypto - returns tuple of date extremities
        """
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
        print(res)
        data = res['prices']
        data = [[dt.datetime.fromtimestamp(x[0]/1000),x[1]] for x in data]
        df = pd.DataFrame(data)
        return df
    
    def write_to_json(self,crypto,currency):
        # NOT WORKING YET - JUST SCRIBBLING
        dates = date_range_lister()
        tracker = {}
        tracker[crypto] = {}
        res = pd.DataFrame()
        for i in tqdm.tqdm(range(len(dates))):
            if i==0: # careful
                pass
            else:
                try:
                    data = self.get_data_on_interval(crypto,currency,dates[i-1],dates[i])
                    res = pd.concat([res,data])
                except json.decoder.JSONDecodeError:
                    print(dates[i],dates[i-1])
                    tracker[crypto][dates[i]]=[dates[i],dates[i-1]]
            time.sleep(1.2)
        return res
    
    def load_from_json(self,crypto):
        with open(f"{crypto}.json",'r') as f_in:
            data = json.loads(json.load(f_in))
        return pd.DataFrame.from_dict(data['data'])

if __name__=="__main__":
    cryptos = ['cardano','usd-coin'] # Change this to the IDs of what you want to retrieve

    gck = Gecko()

    for crypto in tqdm.tqdm(cryptos):
        print('\n',crypto,'\n')

        data = gck.write_to_json(crypto,'usd')
        with open(f'{crypto}.json','w') as f:
            json.dump(data.to_json(orient='split',date_format='iso'),f)        
       
    # xx = gck.write_to_json('ethereum','usd')
    # with open('test_eth_1.json','w') as f:
    #     json.dump(xx.to_json(orient='split',date_format='iso'),f)        
