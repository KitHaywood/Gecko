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
import numpy as np
from scipy.fft import fft
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
try:
    from symfit import parameters, variables, sin, cos, Fit
except ImportError:
    print('please install symfit')


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
    
    def fit_curve_fourier(self,data,window):
        """
        returns: function with fitted_curve
        """
        def fourier_series(x, f, n=0):
            """
            Returns a symbolic fourier series of order `n`.

            :param n: Order of the fourier series.
            :param x: Independent variable
            :param f: Frequency of the fourier series
            """
            # Make the parameter objects for all the terms
            a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)])) # check range 0-->
            sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)])) # check range 1-->
            # Construct the series
            series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                            for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))

            return series

        def fitter(data,window,degree):
            self.window = window
            x,y = variables('x,y') # sets up variable for symfit object
            model_dict = {y: fourier_series(x,4,n=degree)} # required output format 
            data['dates'] = [dt.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S.%fZ") for x in data[0]]
            self.x_data = data['dates'].iloc[-self.window:].apply(lambda x: mdates.date2num(x)) # sort the dates out
            self.y_data = data[1].iloc[-window:] # trim dataset to window size
            fit = Fit(model_dict,x=self.x_data,y=self.y_data) # create fit object
            fit_result = fit.execute()
            y_hat = fit.model(x=self.x_data,**fit_result.params).y
            return fit_result,y_hat
    
        self.res = {}
        for i in range(3,11,1): # Want to optimize on this
            self.res[f"{i}_result"] = fitter(data,250,i)[0]
            self.res[f"{i}_yhat"] = fitter(data,250,i)[1]
        return self.res

    def fourierExtrapolation(self, x, n_predict):
        n = x.size
        n_harm = 50                     # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)         # find linear trend in x
        x_notrend = x - p[0] * t        # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n)              # frequencies
        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
        indexes.reverse()
    
        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t


    def create_coeff_matrix(self,res):
            tester = np.array([np.fromiter(v.params.values(),float) for k,v in res.items() if '_result' in k],dtype=object)
            max_len = max([len(i) for i in tester])
            last_res = np.array([ np.pad(tester[i],
                    (0,max_len-len(tester[i])),
                    'constant',
                        constant_values=0) for i in range(len(tester))]) # got to make A and B matrix
            return last_res


    def plot_result(self):
        if self.window != 0:
            new_res = np.array([self.res[f"{k.split('_')[0]}_yhat"] for k,v in self.res.items()])
            mean = np.mean(new_res,axis=0)         
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            for k,v in self.res.items():
                ax.plot(self.x_data,self.res[f"{k.split('_')[0]}_yhat"],alpha=0.1,color='k')
            ax.plot(self.x_data,self.y_data)
            ax.plot(self.x_data,mean)
            ax.plot(self.x_data,self.y_data.rolling(20).mean())
            ax.grid
        else:
            print('Window must be > 0')
        return fig


def main(window,):
    # main = gck_main() # THIS GETS DATA AND WRITES TO JSON
    s = Strategy()
    data = s.load_data('ethereum')
    fit = s.fit_curve_fourier(data,window)
    coef_mat = s.create_coeff_matrix(fit)    
    s.plot_result()
    return data,coef_mat
    

if __name__=="__main__":
    main(500)
    main(5000)