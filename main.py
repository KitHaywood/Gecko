try:
    from pycoingecko import CoinGeckoAPI
except ImportError:
    print('please install pycoingecko in the CLI')
import datetime as dt
try:
    import pandas as pd
except ImportError: 
     print('please install pandas in the CLI')
import time
import requests
import json
import tqdm
import sys
import os
from utils import date_range_lister
from scipy.optimize import curve_fit
import matplotlib.dates as mdates

# TODO - Write the dox uuuhhhhhh
# TODO - Write func to update data on increment
# TODO - Go back and use market_chart_range to get 
 
class Gecko:
    """
    Class for interaction with CoinGecko API for data retreival and json write
    """
    def __init__(self):
        self.cg = CoinGeckoAPI()
        return None

    def get_all_coins(self):
        """returns list of crypto instruments in CoinGecko"""
        return self.cg.get_coins_list()
    
    def get_market_data(self,symbol,currency,days):
        """
        PARAMETERS: 
        crypto-id(symbol) --> str (see CoinGecko dox for crypto-ids)
        currency          --> str (industry standard for sovereign currency, i.e. 'usd')
        data resoltion    --> '1','14','30','max'
        
        RETURNS: DataFrame
        """
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
        if isinstance(res,pd.DataFrame):
            with open(f'{crypto}.json','w') as f:
                json.dump(res.to_json(orient='split',date_format='iso'),f)
        else:
            print('self.get_data_on_interval returned incorrect datatype')
        return res
    
    def load_from_json(self,crypto):
        """
        Parameters: str --> crypto ID as listed in CoinGeckoAPI
        
        Returns: DataFrame of loaded data from JSON
        
        """
        with open(f"{crypto}.json",'r') as f_in:
            data = json.loads(json.load(f_in))
        return pd.DataFrame.from_dict(data['data'])
    
def gck_main():
    if sys.argv[0]=='main.py':
        if len(sys.argv[1:])>0: # USER DEFINED
            print('\n',f"CLI cryptos passed, getting --> {', '.join([x.upper() for x in list(sys.argv[1:])])}",'\n')
            cryptos = list([x.lower() for x in sys.argv[1:]])
        else:
            cryptos = ['cardano','usd-coin'] # DEFAULTS
            print('\n',"No CLI cryptos passed, using default {} , {} ".format(
                str(cryptos[0]).upper(),
                str(cryptos[1]).upper()
                ),'\n')
    else:
        pass

    gck = Gecko()
    for crypto in tqdm.tqdm(cryptos):
        gck.write_to_json(crypto,'usd')
    return 0    

class Strategy:
    
    def __init__(self) -> None:
        pass

    def load_data(self,crypto):
        with open(f'data\{crypto}.json') as f:
            data = json.loads(json.load(f))
        return pd.DataFrame.from_dict(data['data'])
    
    def fit_curve(self,data,window):
        """
        returns: function with fitted_curve
        """
        def objective(x,a,b,c,d,e,f):
            return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

        popt,_ = curve_fit(objective,mdates.date2num(data[0].iloc[-window:]),data[1].iloc[-window:])
        x_line = np.arange(min(data[0].iloc[-window:]),max(data[0].iloc[-window:]),1)
        yline = objective(x_line,a,b,c,d,e,f)
        return 

    




if __name__=="__main__":
    # main = gck_main()
    # if main==0:
    #     print('\n','Program Executed Successfully','\n')
    s = Strategy()
    data = Strategy().load_data('ethereum')
    print(s.fit_curve(data,50))
