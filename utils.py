import datetime as dt
import pandas as pd

def date_range_lister():
    end = dt.datetime.today()
    diff = dt.timedelta(days=1827, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    start = end - diff
    dates = pd.date_range(start=start,end=end,freq='D')
    return list(dates)

if __name__=="__main__":
    print(date_range_lister())

