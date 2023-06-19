# Fetch 1 year TS data from Screener for all stocks in Master_Scrips

import pandas as pd
import urllib.request 
import requests
import simplejson
import yfinance as yf

from sqlalchemy.orm import sessionmaker
from time import sleep

from DBO.Common import RtIntradayPrice
from Utils.CommonUtils import Screener_Api_url, Screener_Api_url2, engine

_DBconnection = engine.connect()
Session = sessionmaker(bind=engine)
session = Session()

_fetchScrips = "SELECT MS.SCRIP_ID, SCRIP_SCREENER_CODE FROM MASTER_SCRIPS MS" #, ScripsMissingLatestPriceData SC where MS.SCRIP_CODE = SC.SCRIP_CODE " #+ newScrips

_last_update_dt = '2023-06-09' 

try:
    print('Fetching data from Master_Scrips')
    _df = pd.DataFrame(_DBconnection.execute(_fetchScrips), columns=['scrip_id', 'screener_code'])

    for index, row in _df.iterrows():
        print(str(row['scrip_id']))
        _1yr_price_dataUrl = Screener_Api_url + str(row['screener_code']).replace(".0", '') + Screener_Api_url2
        
        response= requests.get(url = _1yr_price_dataUrl)
        
        try:
            # load all the datasets into separat DFs and then merge them
            json_data   = response.json()
            _df_ts_price    = pd.DataFrame.from_dict(json_data['datasets'][0]['values'])
            _df_ts_price.rename(columns={0: 'scrip_id', 1: 'price'}, inplace=True)
            #print(_df_ts_price.columns.to_list())

            _df_ts_volume   = pd.DataFrame.from_dict(json_data['datasets'][1]['values'])
            _df_ts_volume.drop([2], axis=1, inplace=True)     # drop column since delivery data is not available
            _df_ts_volume.rename(columns={0: 'scrip_id', 1: 'volume'}, inplace=True)

            df_merged = _df_ts_price.merge(_df_ts_volume, how='inner', left_on='scrip_id', right_on='scrip_id')

            _df_ts_dma50    = pd.DataFrame.from_dict(json_data['datasets'][2]['values'])
            _df_ts_dma50.rename(columns={0: 'scrip_id', 1: 'dma50'}, inplace=True)

            df_merged = df_merged.merge(_df_ts_dma50, how='inner', left_on='scrip_id', right_on='scrip_id')

            _df_ts_dma200   = pd.DataFrame.from_dict(json_data['datasets'][3]['values'])
            _df_ts_dma200.rename(columns={0: 'scrip_id', 1: 'dma200'}, inplace=True)

            df_merged = df_merged.merge(_df_ts_dma200, how='inner', left_on='scrip_id', right_on='scrip_id')
            print(df_merged)

            for i, dailyrow in df_merged.iterrows():  #Open, High, Low values set in "UpdateOHLC"
                if _last_update_dt < dailyrow.values[0]:   #Add only new data. No updates
                    #print('1')
                    insertTSData = RtIntradayPrice(str(row['scrip_id']), 0,0,0, dailyrow.values[1], 0, 0
                                                    , dailyrow.values[2], 0, dailyrow.values[3], dailyrow.values[4]
                                                    , dailyrow.values[0])
                    session.add(insertTSData)
                    session.commit()

        except simplejson.errors.JSONDecodeError as jsondecodeexception:
            print(jsondecodeexception)
            continue
        
        sleep(2)

except urllib.error.HTTPError as exception:
    print('\n...Unable to fetch: '+ exception) 

session.close()
