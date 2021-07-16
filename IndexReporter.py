# -*- coding: utf-8 -*-
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import json
from nsetools import Nse
#from pprint import pprint
import requests
import pandas as pd
import datetime
from datetime import timedelta
from Indicators import SuperTrend, RSI
import math
import logging
#import redis

#redis_conn = redis.Redis(host='localhost', port=6379, db=0)

bot_token = ''
bot_chatID = '' #Bot
bot_chatID = '' #MarketMessages group

indicesDir = "/Users/guest1/Downloads/bhavcopyDownloads/indices/"

indiceFiles = {'Nifty 50': 'Nifty50', 'Nifty Pharma': 'Nifty_Pharma', 'Nifty Realty': 'Nifty_Realty', 'Nifty Metal': 'Nifty_Metal', 'Nifty IT': 'Nifty_IT', 'Nifty Infrastructure': 'Nifty_Infra', 'Nifty Financial Services': 'Nifty_Fin', 'Nifty Bank': 'Nifty_Bank', 'Nifty Auto': 'Nifty_Auto', 'Nifty Energy': 'Nifty_Energy', 'Nifty FMCG': 'Nifty_FMCG', 'Nifty Commodities': 'Nifty_Commodities'} #, 'Nifty Oil & Gas': 'Nifty_Oil'

_short_indicesDict = {'Nifty Pharma': '^CNXPHARMA', 'Nifty Energy': '^CNXENERGY', 'Nifty Financial Services': '^CNXFIN', 'Nifty Infrastructure': '^CNXINFRA', 'Nifty Metal': '^CNXMETAL', 'Nifty Commodities': '^CNXCMDT', 'Nifty FMCG': '^CNXFMCG', 'Nifty Auto': '^CNXAUTO', 'Nifty Realty': '^CNXREALTY', 'Nifty 50': '^NSEI', 'Nifty Bank': '^NSEBANK', 'Nifty IT': '^CNXIT'}

_indicesDict = {}

today = datetime.date.today()
while today.weekday() >= 5:  # find nearest weekday    
    today = today - timedelta(1)

_Date = today.strftime('%d-%b-%Y')  #'%d%b%Y'
logging.info(_Date)

logsDir = "/Users/guest1/Downloads/bhavcopyDownloads/logs/"

logging.basicConfig(filename=logsDir + 'Index_' + today.strftime('%d%h%Y') + '.log', level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

_Open, _High, _Low, _Close, _Shares, _Turnover = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

_50sma_Dict = {}
_20sma_Dict = {}
_10sma_Dict = {}
_Bol_Dict = {}
_Indicators_Dict = {}
_SuperTrend_Dict = {}
_RSI_Dict = {}
_index_threshold = 5.0

def identifyCandle(prev_open, prev_close, curr_open, curr_close):
    result = []
    # harami
    if prev_open > prev_close:
        if curr_open > prev_close and curr_close < prev_open:
            result.append(' Harami ')
    if prev_open < prev_close:
        if curr_open > prev_open and curr_close < prev_close:
            result.append(' Harami ')

    # check for marubozu

    return result


def updateIndexCsv(file, Date, Open, High, Low, Close, Volume):

    data = [[str(_Date), str(Open), str(High), str(Low), str(Close), str(Volume), "0.0"]]
    df = pd.DataFrame(data)
    
    df_temp = pd.read_csv(indicesDir + file + '.csv')
    current_date = df_temp.tail(1)['Date'].to_string(index=False).strip()

    if current_date < _Date:
        df.to_csv(indicesDir + file + '.csv', mode='a', index=False, header=False)


for _index, file in indiceFiles.items():
    df_index = pd.read_csv(indicesDir + file + '.csv')
    #print(df_index)
    _50sma = (round(df_index['Close'].tail(50).sum()/ 50, 2))
    _50sma_Dict[_index] = _50sma
    _20sma = (round(df_index['Close'].tail(20).sum()/ 20, 2))
    _20sma_Dict[_index] = _20sma
    _10sma = (round(df_index['Close'].tail(10).sum()/ 10, 2))
    _10sma_Dict[_index] = _10sma
    _Bol_Std_Dev = round(df_index['Close'].tail(20).std(), 2)
    _Bol_Dict[_index] = _Bol_Std_Dev
    
    # SuperTrend
    try:
        df_SuperTrend = SuperTrend(df_index, 10, 3)
    except:
        logging.info(' Supertrend calculation error ')
    
    # RSI
    try:
        df_RSI = RSI(df_index)
    except:
        logging.info(' RSI calculation error ')

    _SuperTrend_Dict[_index] = df_RSI.tail(1)['ST_10_3'].to_string(index=False)
    _RSI_Dict[_index] = df_RSI.tail(1)['RSI_14'].to_string(index=False)
    #print(_index)

    #print(df_RSI.tail(1)['ST_10_3'].to_string(index=False))  # this DF contains ATR(10), ST(10,3), and RSI(14)
    _Indicators_Dict[_index] = 'ATR_10:' + df_RSI.tail(1)['ATR_10'].to_string(index=False) + ', SuperTrend:' + df_RSI.tail(1)['STX_10_3'].to_string(index=False) + ',' + df_RSI.tail(1)['ST_10_3'].to_string(index=False) + ', RSI_14:' + df_RSI.tail(1)['RSI_14'].to_string(index=False)


message = ''

for _index, yahooSymbol in _indicesDict.items():
    my_share = share.Share(yahooSymbol)
    logging.info(yahooSymbol)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY, 1, share.FREQUENCY_TYPE_DAY, 1)
    except YahooFinanceError as e:
        logging.info(e.message)

    data = json.dumps(symbol_data)
    data_str = json.loads(data)
    
    if data_str is None:
        continue

    logging.info(data_str)   # Error handling required here
    prev_open = round(data_str['open'][0], 2)
    curr_open = round(data_str['open'][1], 2)
    prev_close = round(data_str['close'][0], 2)
    curr_close = round(data_str['close'][1], 2)

    curr_high = round(data_str['high'][1], 2)
    curr_low = round(data_str['low'][1], 2)

    # Index file update
    # get filename from indiceFiles dictionary
    _file = indiceFiles.get(_index)
    updateIndexCsv(_file, _Date, curr_open, curr_high, curr_low, curr_close, 0)

    message += "\n"
   
    message += str(_index) + " opened " + str(round(100 * (curr_open - prev_close)/ prev_close, 2)) + "%, and is currently " + str(round(100 * (curr_close - prev_close)/ prev_close, 2)) + "% up/down\n"

    candle_pattern = []
    candle_pattern = identifyCandle(prev_open, prev_close, curr_open, curr_close)

    if len(str(candle_pattern)) > 2 :
        message += str(_index) + " has formed a " + str(candle_pattern)

    message += "\n"

    if _index in _Indicators_Dict.keys():
        message += str(_index) + ' is at ' + str(_Indicators_Dict[_index]) + '\n'

    if _index in _50sma_Dict.keys():
        curr_index_state = (curr_close - _50sma_Dict[_index])/ curr_close
        message += str(_index) + ' is at ' + str(round(curr_index_state * 100, 2)) + '% of its 50sma'

    if _index in _10sma_Dict.keys(): 
        curr_index_state = (curr_close - _10sma_Dict[_index])/ curr_close
        message += ' and ' + str(round(curr_index_state * 100, 2)) + '% of its 10sma\n'

    if _index in _20sma_Dict.keys(): 
        curr_index_state = (curr_close - _20sma_Dict[_index])/ curr_close
        message += str(_index) + ' is at ' + str(round(curr_index_state * 100, 2)) +'% of its 20sma'
        if _index in _Bol_Dict.keys():
            boll_U = _20sma_Dict[_index] + 2 * _Bol_Dict[_index]
            boll_L = _20sma_Dict[_index] - 2 * _Bol_Dict[_index]
            boll_U_Diff = (curr_close - boll_U)/ curr_close
            boll_L_Diff = (curr_close - boll_L)/ curr_close
            message += ' and ' + str(round(boll_U_Diff * 100, 2)) + '% of its Boll_U and ' + str(round(boll_L_Diff * 100, 2)) + '% of its Boll_L '+'\n'

# yahoo_finance_api2 doesn't return the same data for all indices... ???????
for _index, yahooSymbol in _short_indicesDict.items():
    my_share = share.Share(yahooSymbol)
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY, 1, share.FREQUENCY_TYPE_DAY, 1)
    except YahooFinanceError as e:
        logging.info(e.message)

    data = json.dumps(symbol_data)
    data_str = json.loads(data)

    if data_str is  None:
        continue
    logging.info(data_str)
    
    curr_open = round(data_str['open'][0], 2)
    curr_close = round(data_str['close'][0], 2) 
    curr_high = round(data_str['high'][0], 2) 
    curr_low = round(data_str['low'][0], 2) 

    marubozu_threshold = 0.05
    #if (curr_open - curr_low + curr_high - curr_close):

    _file = indiceFiles.get(_index)
    updateIndexCsv(_file, _Date, curr_open, curr_high, curr_low, curr_close, 0)

    message += "\n\n"
    message += str(_index) + " opened at " + str(curr_open) + ", and is currently " + str(curr_close) + ", moved " + str(round(100 * (curr_close - curr_open)/ curr_open, 2)) + "%"

    message += "\n"

    if _index in _Indicators_Dict.keys():
        message += str(_index) + ' is at ' + str(_Indicators_Dict[_index]) + '\n'

    if _index in _50sma_Dict.keys(): 
        curr_index_state = (curr_close - _50sma_Dict[_index])/ curr_close
        message += str(_index) + ' is at ' + str(round(curr_index_state * 100, 2)) + '% of its 50sma'

    if _index in _10sma_Dict.keys(): 
        curr_index_state = (curr_close - _10sma_Dict[_index])/ curr_close
        message += ' and ' + str(round(curr_index_state * 100, 2)) + '% of its 10sma\n'

    if _index in _20sma_Dict.keys(): 
        curr_index_state = (curr_close - _20sma_Dict[_index])/ curr_close
        message += str(_index) + ' is at ' + str(round(curr_index_state * 100, 2)) +'% of its 20sma'
        if _index in _Bol_Dict.keys():
            boll_U = _20sma_Dict[_index] + 2 * _Bol_Dict[_index]
            boll_L = _20sma_Dict[_index] - 2 * _Bol_Dict[_index]
            boll_U_Diff = (curr_close - boll_U)/ curr_close
            boll_L_Diff = (curr_close - boll_L)/ curr_close
            message += ' and ' + str(round(boll_U_Diff * 100, 2)) + '% of its Boll_U and ' + str(round(boll_L_Diff * 100, 2)) + '% of its Boll_L '+'\n'
 
    if _index in _SuperTrend_Dict.keys(): 
        #print(_index)
        #print(float(_SuperTrend_Dict[_index]))
        curr_index_state = (curr_close - float(_SuperTrend_Dict[_index]))/ curr_close
        message += str(_index) + ' is at ' + str(round(curr_index_state * 100, 2)) + '% of its SuperTrend'

logging.info(message)
print(message)

send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&text=' + message
resp = requests.get(send_text)
logging.info(resp)

"""
nse = Nse()
adv_dec = nse.get_advances_declines()
#pprint(adv_dec)

top_gainers = nse.get_top_gainers()

message += "\n\n"
message += 'Current Nifty 50 top gainers/ losers '

i = 0
for dict in top_gainers:
    string = ''
    for key, value in dict.items():
        if key == 'symbol' or key == 'netPrice' or key == 'ltp':
            if key == 'ltp':
                string += str(value) + ' ('
            if key == 'netPrice':
                string += str(value) + '%)' 
            elif key == 'symbol':
                string += str(value.replace('&','_') + ', ')
    message += "\n"
    message += string

"""
