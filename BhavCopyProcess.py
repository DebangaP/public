# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:27:54 2020

@author: @debanga
"""

# In[14]:

import urllib.request
import os
import datetime
import numpy as np
from datetime import timedelta
import zipfile
import pandas as pd
import requests
import dataframe_image as dfi
import yahoo_fin.stock_info as si
from tqdm import tqdm
from Lists import excStockslist, trading_holidays

pd.options.mode.chained_assignment = None  # default='warn'

    #To do: Monthly folder
downloadDir = "/Users/guest1/Downloads/bhavcopyDownloads/"

nse_base_url = "https://www1.nseindia.com/content/historical/EQUITIES/"
bse_base_url = "https://www.bseindia.com/download/BhavCopy/Equity/"

marketCap_file = "MCAP_31032020_0.csv"

bot_token = '1359519581:AAGd60acQuItRGsICUHFTIC5m-_QhYFSXQ0'
bot_chatID = '1053026020' #Bot  
bot_chatID = '-387605433' #MarketMessages group

#trading_holidays = ['16Nov2020', '30Nov2020', '25Dec2020']

lookback_days = 1 # To do...compare with last 'n' days and check for Volume and Price spikes
num_rows = 20
volFilter = 50000
price_low = 51
price_high = 5000
bollinger_threshold = 5 # percentage
_20Sma_threshold = 0.1 # percentage
stop_loss = 0.0125   # percentage, in decimals
profit = 0.025       # percentage, in decimals
buy_range = 0.01    # percentage, in decimals

today = datetime.date.today()
process_date = today

now = datetime.datetime.now()
if (now.hour < 17):
    print('\n...too early, current day file may not be available...processing for previous day')
    process_date = process_date - timedelta(1)

# find nearest weekday    
while process_date.weekday() >= 5:
    process_date = process_date - timedelta(1)
    
#process_date = today - timedelta(1)
print(process_date)

telegram_m1 = ''
telegram_m2 = ''

bse_date = process_date.strftime('%d-%m-%y').replace("-","")
bse_url = bse_base_url + "EQ_ISINCODE_" + bse_date + ".zip"

nse_date = process_date.strftime('%d-%h-%Y').replace("-","").upper()
nse_file_name = "cm" + nse_date + "bhav.csv"
nse_short_str = process_date.strftime('%Y') + "/" + process_date.strftime('%h').upper() + "/" + nse_file_name
nse_url = nse_base_url + nse_short_str + ".zip"

def downloadBhavCopy(exchange, fileUrl, saveAsFile):
    print ("\n...Starting download of bhavcopy from "+fileUrl)
    
    user_agent = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Mobile Safari/537.36'
    headers={'User-Agent':user_agent,} 
    request=urllib.request.Request(fileUrl, None, headers) #The assembled request
    
    try:
        response = urllib.request.urlopen(request)
        with open(saveAsFile, 'wb') as f:
            f.write(response.read())
    except urllib.error.HTTPError as exception:
        print('\n...Current file not available for '+ exchange)


def unZip(zipFile, directory_to_extract_to):
    print("\nExtracting zip archive to " + directory_to_extract_to)
    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    

def mergeOnISIN():
    print("\n...Start merging files based on ISIN -> Adding up volumes, and filtering")
    try:
        df_bse = pd.read_csv(downloadDir + "dataDump/EQ_ISINCODE_" + bse_date + ".csv")
        df_nse = pd.read_csv(downloadDir + "dataDump/" + nse_file_name)
        #df_excludeETF = d.DataFrame(excStockslist, columns=['SYMBOL'])
    
        #..Merge/ Inner Join on ISIN from NSE BhavCopy and ISIN_CODE from BSE BhavCopy
        df_merged = df_nse.merge(df_bse, how='inner', left_on='ISIN', right_on='ISIN_CODE')
        df_merged['TOTAL_VOLUME'] = df_merged['TOTTRDQTY'] + df_merged['NO_OF_SHRS']
        df_merged['%GAIN'] = round(100 * (df_merged['CLOSE_x'] - df_merged['PREVCLOSE_x'])/df_merged['PREVCLOSE_x'],2)
        
        df_merged.drop(df_merged[df_merged['CLOSE_x'] < price_low].index, inplace=True)
        df_merged.drop(df_merged[df_merged['CLOSE_x'] > price_high].index, inplace=True)
        df_merged.drop(df_merged[df_merged['TOTAL_VOLUME'] < volFilter].index, inplace=True) #makes a big difference
        df_merged.drop(df_merged[df_merged['SERIES'] != 'EQ'].index, inplace=True)
        #df_merged.drop(df_merged[df_merged['%GAIN'] < 0].index, inplace=True) # this causes missing price data in "Vol_gainers"
 
        #df_merged.info(verbose=True)
        df_merged = df_merged[~df_merged.SYMBOL.isin(excStockslist)]
        
        df_merged.sort_values(by=['%GAIN'], ascending=False, inplace=True)
        #df_merged.info(verbose=True)

        # Save only selected columns to CSV file -- below file may not be needed
        header = ["SYMBOL", "ISIN", "SC_NAME", "OPEN_x", "HIGH_x", "LOW_x", "CLOSE_x", "PREVCLOSE_x", "TOTTRDQTY", "NO_OF_SHRS", "TOTAL_VOLUME", "%GAIN"]
        df_merged.to_csv(downloadDir + "Merged"+ bse_date + ".csv", columns = header)
    except FileNotFoundError as e:
        print("\n...Error...One of the files is not available for merging")
        
        
def plotImageMessage(dft, date1):
    print('\nPlotting image for message')
    caption = ' Buy reco based on EOD of ' + str(date1.strftime('%d-%b-%Y')) + " (valid for next trading day) _BB_"

    dft1 = dft.style.set_caption(caption).background_gradient().set_precision(2)
    dfi.export(dft1, downloadDir + '/images/MessageImage' + str(date1) + '.jpg')  


def plotImage(dft, date1):
    print('\nInside plot image')  
    #print(dft)      
    
    caption = 'Buy recommendations based on EOD of ' + str(date1.strftime('%d-%b-%Y')) + ' (comparison with -' + str(lookback_days) + ' days)/ {Market cap as of 31-March-2020}'

    dft = dft.astype({"Open": float, "Close": float, "%Gain": float,  "Volume": float, "%Vol_Gain": float}) # to correct for image gradient issues (50sma is '-' sometimes)
    
    dft.drop(dft[dft['Signal'] != 'Buy'].index, inplace=True)
    
    #dft1 = dft.style.hide_index().set_caption(caption).background_gradient().set_precision(2)
    dft1 = dft.style.set_caption(caption).background_gradient().set_precision(2)
    dfi.export(dft1, downloadDir + '/images/Image' + str(date1) + '.jpg')
    
    print('\nInside plot image')  
    
    
    
def buySignal(row):
    
    if row['Symbol'] == 'UTIAMC':
        return
    
    L_Wick = row['Open'] - row['Low']
    U_Wick = row['High'] - row['Close']
    Candle_body = row['Close'] -  row['Open']
    
    Signal = ''
    
    if L_Wick > U_Wick:
        Signal = 'Buy'

    # green Doji
    #print(row['Symbol'])
    #if (row['Close'] - row['Open'])/ row['Open'] <= 0.5:
    #    Signal = 'Buy'
        
    if (U_Wick == 0 and L_Wick == 0 and Candle_body > 5): # Marubozu check -- needs change
        Signal = 'Buy'
        
    # walking the Bollinger upper band        
    if abs(100 * (row['Close'] - row['Boll_U'])/row['Close']) <= bollinger_threshold:
        Signal = 'Buy'
        if row['Symbol'] == 'WOCKPHARMA':
            print('-- Buy --')


    # Walking the Bollinger upper band, and if there still might be upside        
    if (row['Close'] >= row['Boll_U'] and row['10sma'] >= row['20sma']):   # should it be 10sma< 20sma????
        Signal = 'Buy'
                
    """
    # Avoiding long upper wick, but allow larger candle body     
    if (U_Wick > 0 and Candle_body > 0):    
        U_Wick_percentage = U_Wick/ Candle_body

        if U_Wick_percentage <= 0.25:
            Signal = 'Buy'
        #elif ((Candle_body > U_Wick) and abs(row['Close'] - row['Boll_U'])/row['Close'] <= 0.25):
        #    Signal = 'Buy'
        #else:
        #    Signal = ''
    """

    # this section new
    if (row['Close'] > row['Boll_U']) and (Candle_body > U_Wick):
        Signal = 'Buy'
    
    ##--> this one cause multiple misses on 17Nov2020
    """
    if row['Open'] > row['Close']:      
        Signal = ''
        print('Signal changed from Buy--' + row['Symbol'])
    """

    # Near the Bollinger lower band, and recovering...
    if abs(100 * (row['Close'] - row['Boll_L'])/row['Close']) <= bollinger_threshold:
        #if abs(100 * (row['Close'] - float(row['50sma']))/row['Close']) <= bollinger_threshold:
            #print((row['Symbol']))
        Signal = 'Buy'

    # ...too far from 20sma...##--> this one cause multiple misses on 17Nov2020
    """
    if (Signal == 'Buy') and ((row['Close'] - row['20sma'])/row['Close'] > _20Sma_threshold * 1.2):
        Signal = ''
        print('Signal changed from Buy--' + row['Symbol'])
    """
    
    
    if row['Symbol'] == 'WOCKPHARMA' or row['Symbol'] == 'LUPIN':
        print(row['Symbol'])
        print(row['Boll_L'])
        #print(abs(100 * (row['Close'] - row['Boll_U'])/row['Close']))
        print(Signal)

    return Signal



"""

    # Looking for long lower wick (possible negative day), but still walking the upper Bollinger band
    if (L_Wick > 0 and abs(Candle_body) > 0 and abs((row['Close'] - row['Boll_U'])/row['Close']) < 0.02):    
        L_Wick_percentage = L_Wick/ abs(Candle_body)  # check the division, sometimes wicks are zero
        
        if L_Wick_percentage >= 0.40:
            Signal = 'Buy'
        else:
            Signal = ''

    # Avoid those with unclear direction
    if U_Wick > 0 and Candle_body < 2:
        if abs (L_Wick - U_Wick)/ U_Wick > 0.05 :
            Signal = ''
                
"""
    
    
def findNdaysVolumePriceSpikes(startDate, prevDate):
    toDate = startDate.strftime('%d-%m-%y').replace("-","")
    fromDate = prevDate.strftime('%d-%m-%y').replace("-","")
    
    df0 = pd.read_csv(downloadDir + "Merged"+ toDate + ".csv")
    df1 = pd.read_csv(downloadDir + "Merged"+ fromDate + ".csv")
    
    df_final = df0.merge(df1, how='inner', on='ISIN')
    
    #df_final.info(verbose=True)
    print(' -xo -xo -')
    print('ASTRAL' in df_final.SYMBOL_x.values)
    
    df_final['%Vol_Gain'] = round(100 * (df_final['TOTAL_VOLUME_x'] - df_final['TOTAL_VOLUME_y'])/df_final['TOTAL_VOLUME_y'],2)
    
    df_final.rename(columns={'SYMBOL_x': 'Symbol', 'SC_NAME_x': 'Name', 'OPEN_x_x': 'Open', 'CLOSE_x_x': 'Close', 'PREVCLOSE_x_x': 'P1_Close', 'PREVCLOSE_x_y': 'P2_Close', 'TOTAL_VOLUME_x': 'Volume', 'TOTAL_VOLUME_y': 'P1_Volume', '%GAIN_x': '%Gain', 'HIGH_x_x': 'High', 'LOW_x_x':'Low'}, inplace=True)

    df_final.sort_values(['%Vol_Gain', '%Gain'], ascending=(False, False), inplace=True)
    #df_final.info(verbose=True)
    
    df_final.info(verbose=True)
    df_tele0 = df_final[['Symbol', 'Open', 'Close', '%Gain', 'Volume', '%Vol_Gain', 'High', 'Low']]
    #df_tele0.reset_index(drop=True, inplace=True)
    #df_final.info(verbose=True)
    #df_tele0.info(verbose=True)
    
    #...add 10/20/50sma from pre-calculated file
    sma_df = pd.read_csv(downloadDir + "Final_file_10_20_50sma_"+ str((process_date).strftime('%d%b%Y')) + ".csv")
    
    df_tele0.drop_duplicates()
    
    df_tele = df_tele0.merge(sma_df[['10sma', '20sma', '50sma', 'Symbol']], how='outer', on='Symbol')
    df_tele.sort_values(['%Gain', '%Vol_Gain'], ascending=(False, False), inplace=True)
    
    # read the Bollinger file and get Stddev
    df_temp_bol = pd.read_csv(downloadDir + "Temp_Bol.csv")

    df_tele = df_tele.merge(df_temp_bol[['stddev', 'Symbol']], how='outer', on='Symbol')
    
    df_tele['Boll_U'] = df_tele['20sma'] + 2 * df_tele['stddev']
    df_tele['Boll_L'] = df_tele['20sma'] - 2 * df_tele['stddev']
    
    df_tele = df_tele.drop(columns=['stddev'])    # drop un-necessary columns

    print(' -xo -xo -')
    print('LUPIN' in df_tele.Symbol.values)

    # Buy Signal calculation done in "buySignal" function
    df_tele['Signal'] = df_tele.apply (lambda row: buySignal(row), axis=1)
    
    print(' -xo -xo -')
    print('LUPIN' in df_tele.Symbol.values)

    # End - Buy Signal calculation
        
    df_tele.drop(df_tele[df_tele['Signal'] != 'Buy'].index, inplace=True)
    df_tele.sort_values(['Signal', '%Vol_Gain'], ascending=(False, False), inplace=True)

    df_tele.to_csv(downloadDir + "Vol_gainers_" + str(startDate.strftime('%d%b%Y')) + ".csv")

    #df_image = df_tele.head(num_rows)    
    df_image = df_tele.dropna()
    
    #df_image.fillna("0", inplace = True)
    
    #commented to get good data from Yahoo Finance
    # One-time only
    #df_mcap0 = pd.read_csv(downloadDir + marketCap_file)
    #df_mcap0['Date'] = '31Mar2020'
    #df_mcap0.to_csv(downloadDir + marketCap_file)
    
    df_mcap = pd.read_csv(downloadDir + marketCap_file)
    df_image1 = df_image.merge(df_mcap[['Symbol', 'MCap(lakhs)', 'Date']], how='outer', on='Symbol')
    
    #print(df_image1)
    df_image1['MCap(lakhs)'] = df_image1['MCap(lakhs)'].str.replace(',', '').astype(float)
    #print(df_image1)
    
    """
    # updated Market-cap data from Yahoo Finance
    #df_image['MCap(lakhs)'] = 0.0
    mcap_LK = 0.0
    i = 0
    
    print(len(df_image1.index))
    
    for i in df_image1.index: 
        if df_image1['Date'][i] == '31Mar2020':
 
            print(str(df_image1['Symbol'][i]) + '.NS')
            mcap_B = si.get_quote_table(str(df_image1['Symbol'][i]) + '.NS')["Market Cap"]  
            print(mcap_B)
        
            if str(mcap_B[-1:]) == 'T':
                mcap_LK = round(10000000 * float(mcap_B[:-1]), 2)
            if str(mcap_B[-1:]) == 'B':
                mcap_LK = round(10000 * float(mcap_B[:-1]), 2)
            #print(mcap_LK)
            df_image1['MCap(lakhs)'][i] = mcap_LK
            df_image1['Date'][i] = str(startDate.strftime('%d%b%Y'))
        i += 1

    print(df_image1)
    #df_image1 = df_image
 
    df_mcap1 = df_mcap.merge(df_image1[['Symbol', 'Date', 'MCap(lakhs)']], how='outer', on='Symbol')
    print(df_mcap1)
    df_mcap1.to_csv(downloadDir + 'MCAP_31032020_1.csv')
    """
    
    
    df_image1.drop(df_image1[df_image1['Signal'] != 'Buy'].index, inplace=True)
    
    df_image1.sort_values(['MCap(lakhs)', '%Vol_Gain'], ascending=(False, False), inplace=True)

    #df_image1.drop(df_image1[df_image1['10sma'] == 0].index, inplace=True)

    df_image1.drop_duplicates()
    df_image1.index = np.arange(1, len(df_image1) + 1) # Reset index to 1
        
    plotImage(df_image1.head(num_rows), startDate) #...saving the message attachment as JPEG

    global telegram_m2
    df_tele.index = np.arange(1, len(df_tele) + 1) # Reset index to 1
    telegram_m2 = str(df_tele.loc[:, ['Symbol', 'Open', 'Close', '%Gain', '%Vol_Gain']])
        
    header = ["Symbol", "ISIN", "Name", "Open", "Close", "P1_Close", "P2_Close", "%Gain", "Volume", "P1_Volume", "%Vol_Gain"]
    #df_final.to_csv(downloadDir + "Merged_final"+ str(startDate).replace("-","") + ".csv", columns = header)
    
    
        # TO DO: Change sort order here to differentiate Vol_ and Bol_
    df_image1.sort_values(['%Vol_Gain'], ascending=(False), inplace=True)
    df_image1.to_csv(downloadDir + "Merged_final"+ str(startDate).replace("-","") + ".csv")


    #Generate Message -- Symbol, Buy_price, Target, Stop-loss, Confidence
    df_short_message = df_image1 #.head(num_rows)
    
    df_short_message.drop(df_short_message[df_short_message['Boll_U'] == 0].index, inplace=True)

    df_short_message = df_short_message.drop(columns=['Open', '%Gain', 'Volume','%Vol_Gain','10sma','20sma','50sma','Boll_L','Boll_U','Signal'])    # drop un-necessary columns


    df_short_message['Message'] = "Buy " + df_image1['Symbol'].astype(str) + " @" + round(df_image1['Close'].astype(float),2).astype(str) + " - " + round(df_image1['Close'].astype(float) * (1 + buy_range),2).astype(str) + ", Stop-loss " 
                    
    df_short_message['Message'] += round(df_image1['Close'].astype(float) * (1 - stop_loss),2).astype(str) + ", Target "+ round(df_image1['Close'].astype(float) * (1 + profit),2).astype(str) + " "
    
    df_short_message.reset_index(drop=True, inplace=True)
    df_short_message.info(verbose=True)
    
    
    
    df_short_message.rename(columns={'Message': 'Recommendation'}, inplace=True)
    df_short_message = df_short_message.drop(columns=['High','Low', 'Date'])
    
    df_short_message.index = np.arange(1, len(df_short_message) + 1) # Reset index to 1
    
    df_short_message.info(verbose=True)
    
    plotImageMessage(df_short_message.head(20), startDate)


    # Changed sort order here to differentiate Vol_ and Bol_
    df_short_message.sort_values(['MCap(lakhs)'], ascending=(False), inplace=True)  
    
    df_short_message.to_csv(downloadDir + "Bollinger_reco_"+ str(startDate).replace("-","") + ".csv")


def calcLastPerf(process_date, toDate):
    df_last_reco = pd.read_csv(downloadDir + "Merged_final"+ str(toDate).replace("-","") + ".csv")
    print(toDate)
    print(bse_date)
    
    df_merged = pd.read_csv(downloadDir + "Merged"+ bse_date + ".csv")
    
    df_final = df_last_reco.merge(df_merged[['SYMBOL', '%GAIN', 'CLOSE_x']], how='inner', left_on='Symbol', right_on='SYMBOL')

    df_final1 = df_final[['Symbol', 'Close', 'CLOSE_x', '%GAIN']]
    
    condlist = [df_final1['%GAIN'] >= 100*profit, df_final1['%GAIN'] <= -100*stop_loss, df_final1['%GAIN'] >= 0, df_final1['%GAIN'] <= 0]
    choicelist = ['Target Achieved', 'Stop-loss hit', 'Profit', 'Loss']
    
    df_final1['Target'] = np.select(condlist, choicelist)
    #df_final1.info(verbose=True)

    df_final1.rename(columns={'Close': 'Yest_Close', 'CLOSE_x': 'Today_Close', '%GAIN': '%Gain'}, inplace=True)

    caption = ' Performance of BUY reco on ' + str(toDate.strftime('%d-%b-%Y')) + ' _BB_ ' 
    #+ '{Net = ' + str(round(df_final1['%Gain'].sum(),2)) + '%} '

    dft1 = df_final1.head(20).style.set_caption(caption).background_gradient().set_precision(2)
    dfi.export(dft1, downloadDir + 'images/Reco_Perf' + str(toDate) + '.jpg')
    
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&text=' + ''
    
    files = {'photo': open(downloadDir + 'images/Reco_Perf' + str(toDate) + '.jpg', 'rb')}    
    send_photo = 'https://api.telegram.org/bot' + bot_token + '/sendPhoto?chat_id=' + bot_chatID

    requests.get(send_text)
    requests.post(send_photo, files=files)
    

def sendTelegramMessage(message, date1):
    print('\n...sending message..')
    
    #send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + message
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&text=' + message
    
    files = {'photo': open(downloadDir + 'images/MessageImage' + str(date1) + '.jpg', 'rb')}    
    send_photo = 'https://api.telegram.org/bot' + bot_token + '/sendPhoto?chat_id=' + bot_chatID

    requests.get(send_text)
    requests.post(send_photo, files=files)
    

#...start of processing logic...
start_time = datetime.datetime.now()
print("\n...processing started...")
print('\n...Run date :'+ str(process_date))
print('...Lookback till :'+ str(process_date - timedelta(lookback_days)))

print('..Process....')
    
if not os.path.exists(downloadDir + "dataDump/EQ_ISINCODE_" + bse_date + ".zip"):
    downloadBhavCopy("BSE", bse_url, downloadDir + "dataDump/EQ_ISINCODE_" + bse_date + ".zip")
    unZip(downloadDir + "dataDump/EQ_ISINCODE_" + bse_date + ".zip", downloadDir + "dataDump/")
else:
    print("\n...BSE BhavCopy found...not downloading...")

if not os.path.exists(downloadDir + "dataDump/" + nse_file_name + ".zip"):
    downloadBhavCopy("NSE", nse_url, downloadDir + "dataDump/" + nse_file_name + ".zip")
    unZip(downloadDir + "dataDump/" + nse_file_name + ".zip", downloadDir + "dataDump/")
else:
    print("\n...NSE BhavCopy found...not downloading...")

mergeOnISIN()
    
toDate = process_date - timedelta(1)
if toDate.strftime('%d%b%Y') in trading_holidays:
    toDate = toDate - timedelta(1)
    
print(toDate)


while (toDate.weekday() >= 5):    # if previous date falls on weekend, move it to previous weekday
    toDate = toDate - timedelta(1)

findNdaysVolumePriceSpikes(process_date, toDate)


telegram_m1 = 'Hello, Here are top Buy recommendations for ' + str(process_date.strftime('%d-%b-%Y'))
sendTelegramMessage(telegram_m1, process_date)

calcLastPerf(process_date, toDate)

end_time = datetime.datetime.now()
print('\n...Ran in ' + str(end_time - start_time) + ' secs flat...')


