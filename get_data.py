import pandas as pd
from datetime import datetime, timezone, timedelta
import calendar
import pandas as pd
import requests
from typing import *
import time

class BinanceClient:
    def __init__(self, futures=False):
        self.exchange = "BINANCE"
        self.futures = futures

        if self.futures:
            self._base_url = "https://fapi.binance.com"
        else:
            self._base_url = "https://api.binance.com"

        self.symbols = self._get_symbols()

    def _make_request(self, endpoint: str, query_parameters: Dict):
        try:
            response = requests.get(self._base_url + endpoint, params=query_parameters)
        except Exception as e:
            print("Connection error while making request to %s: %s", endpoint, e)
            return None

        if response.status_code == 200:
            return response.json()
        else:
            print("Error while making request to %s: %s (status code = %s)",
                         endpoint, response.json(), response.status_code)
            return None

    def _get_symbols(self) -> List[str]:

        params = dict()

        endpoint = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
        data = self._make_request(endpoint, params)

        symbols = [x["symbol"] for x in data["symbols"]]

        return symbols

    def get_historical_data(self, symbol: str, interval: Optional[str] = "1s", start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = 1500):

        params = dict() 

        params["symbol"] = symbol # Coin name
        params["interval"] = interval # 1s
        params["limit"] = limit # 1500

        if start_time is not None: 
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        
        endpoint = "/fapi/v1/klines" if self.futures else "/api/v3/klines"

        raw_candles = self._make_request(endpoint, params)

        candles = []

        if raw_candles is not None: 
            for c in raw_candles:
                candles.append((float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]),))
            return candles
        else:
            return None

def ms_to_dt_utc(ms: int) -> datetime:
    return datetime.utcfromtimestamp(ms / 1000)

def ms_to_dt_local(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000)

def GetDataFrame(data):
    df = pd.DataFrame(data, columns=['Timestamp', "Open", "High", "Low", "Close", "Volume"])

    df["Timestamp"] = df["Timestamp"].apply(lambda x: ms_to_dt_local(x))

    df['Date'] = df["Timestamp"].dt.strftime("%d/%m/%Y")

    df['Time'] = df["Timestamp"].dt.strftime("%H:%M:%S")

    column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]

    df = df.set_index('Timestamp')

    df = df.reindex(columns=column_names)

    return df

def GetHistoricalData(client, symbol, start_time, end_time, limit=1500):
    collection = []

    while start_time < end_time:
        data = client.get_historical_data(symbol, start_time=start_time, end_time=end_time, limit=limit) #list of tuples
        print(client.exchange + " " + symbol + " : Collected " + str(len(data)) + " initial data from "+ str(ms_to_dt_local(data[0][0])) +" to " + str(ms_to_dt_local(data[-1][0])))
        start_time = int(data[-1][0] + 1000)
        collection +=data
        time.sleep(1.1)

    return collection


def get_data(coin, time):
    client = BinanceClient(futures=False)
    symbol = coin # coin name
    now = datetime.now() # 2020-12-01 12:00:00  
    pretimestem = int(round(now.timestamp())) - time # timestamp 30s before
    from_time = datetime.fromtimestamp(pretimestem)

    current_time = now.strftime("%Y-%m-%d %H:%M:%S") # current time 2020-12-01 12:00:00  

    from_time = from_time.strftime("%Y-%m-%d %H:%M:%S") # 30s  before 2020-12-01 11:59:30

    fromDate = int(datetime.strptime(from_time, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    toDate = int(datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

    data = GetHistoricalData(client, symbol, fromDate, toDate) # list of data from 30s before to current time

    df = GetDataFrame(data)

    return df # dataframe of data in 30s before to current time

def get_klines_iter(symbol, interval, start, end = None, limit=1000):

    df = pd.DataFrame()

    if start is None:
        print('start time must not be None')
        return
    start = calendar.timegm(datetime.fromisoformat(start).timetuple()) * 1000

    if end is None:
        dt = datetime.now(timezone.utc)
        utc_time = dt.replace(tzinfo=timezone.utc)
        end = int(utc_time.timestamp()) * 1000
        return
    else:
        end = calendar.timegm(datetime.fromisoformat(end).timetuple()) * 1000
    last_time = None

    while len(df) == 0 or (last_time is not None and last_time < end):
        url = 'https://api.binance.com/api/v3/klines?symbol=' + \
              symbol + '&interval=' + interval + '&limit=1000'
        if(len(df) == 0):
            url += '&startTime=' + str(start)
        else:
            url += '&startTime=' + str(last_time)

        url += '&endTime=' + str(end)
        df2 = pd.read_json(url)
        df2.columns = ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Closetime',
                       'Quote asset volume', 'Number of trades', 'Taker by base', 'Taker buy quote', 'Ignore']
        dftmp = pd.DataFrame()
        dftmp = pd.concat([df2, dftmp], axis=0, ignore_index=True, keys=None)

        dftmp.Opentime = pd.to_datetime(dftmp.Opentime, unit='ms')
        dftmp['Date'] = dftmp.Opentime.dt.strftime("%d/%m/%Y")
        dftmp['Time'] = dftmp.Opentime.dt.strftime("%H:%M:%S")
        dftmp = dftmp.drop(['Quote asset volume', 'Closetime', 'Opentime',
                      'Number of trades', 'Taker by base', 'Taker buy quote', 'Ignore'], axis=1)
        column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
        dftmp.reset_index(drop=True, inplace=True)
        dftmp = dftmp.reindex(columns=column_names)
        string_dt = str(dftmp['Date'][len(dftmp) - 1]) + 'T' + str(dftmp['Time'][len(dftmp) - 1]) + '.000Z'
        utc_last_time = datetime.strptime(string_dt, "%d/%m/%YT%H:%M:%S.%fZ")
        last_time = (utc_last_time - datetime(1970, 1, 1)) // timedelta(milliseconds=1)
        df = pd.concat([df, dftmp], axis=0, ignore_index=True, keys=None)
        # Drop the 'Time' column
        df = df.drop(columns=['Time'])
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.set_index('Date')
        # df.to_csv('0y_eth_only17andnew.csv', index=True, header=True)
    return df