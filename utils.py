import datetime as dt
import pandas as pd

def date_range_lister():
    end = dt.date.today()
    diff = dt.timedelta(days=1827, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    start = end - diff
    dates = pd.date_range(start=start,end=end,freq='D')
    dates = [int(x.timestamp()) for x in dates]
    return list(dates)

def objective(x,a,b,c,d):
    return (a * x) + (b * x**2) + (c * x**3) + d

def obj2(x,a,b,d):
    return a * np.exp((-x/b)) + d
    
if __name__=="__main__":
    print(date_range_lister())

