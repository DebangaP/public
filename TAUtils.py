# -*- coding: utf-8 -*-
import redis
import pandas as pd
import datetime
import json
import pickle
import zlib
from Indicators import SuperTrend, RSI
import urllib.request

# connect to local Redis instance --> all API calls to be stored in Redis which can be then exported

"""
redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

data = [['tom', 10.11111], ['nick', 15], ['juli', 14.0]]   
df_tautils = pd.DataFrame(data, columns = ['Name', 'Age']) 
"""

def downloadBhavCopy(exchange, fileUrl, saveAsFile):
    print ("\n...Starting download of bhavcopy from "+fileUrl)
    
    user_agent = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Mobile Safari/537.36'
    headers={'User-Agent':user_agent,} 
    request=urllib.request.Request(fileUrl, None, headers) #The assembled request
    
    try:
        response = urllib.request.urlopen(request)
        with open(saveAsFile, 'wb') as f:
            f.write(response.read())
        print(' download completed ')
    except urllib.error.HTTPError as exception:
        print('\n...Current file not available for '+ exchange)


def calculateSuperTrend_RSI(scrip):
    downloadDir = "/Users/guest1/Downloads/"

    fileUrl = 'https://www1.nseindia.com/live_market/dynaContent/live_watch/get_quote/getHistoricalData.jsp?symbol=' + scrip + '&series=EQ&fromDate=01Nov2020&toDate=01Dec2020'

    downloadBhavCopy('NSE', fileUrl, downloadDir + scrip + '.csv')

    df_index = pd.read_csv(downloadDir + scrip + '.csv', usecols=[0,3,4,5,7,8,9])

    df_index = df_index[::-1]    # reverse the sort order

    try:
        df_SuperTrend = SuperTrend(df_index, 10, 3)
    except:
        print(' Supertrend calculation error ')

    try:
        df_RSI = RSI(df_index)
        print(df_RSI.tail(20))
    except:
        print(' RSI calculation error ')

    return df_RSI.tail(5)


calculateSuperTrend_RSI('INFY')


def saveToRedis(key, Dataframe, **kwargs):
    #redis_conn.set(key, df.to_json())   # doesn't store decimals ... 
    redis_conn.set(key, zlib.compress(pickle.dumps(df)))    # stores decimals

def getFromRedis(key):
    #temp_json = redis_conn.get(key)
    #df0 = pd.read_json(temp_json)
    df0 = pickle.loads(zlib.decompress(redis_conn.get(key)))
    return df0

def test():
    start_time = datetime.datetime.now()
    saveToRedis('df', df)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    print(getFromRedis('df'))
    end_time = datetime.datetime.now()
    print(end_time - start_time)

