import csv
import time
import pandas as pd
import math
from datetime import datetime

timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)

workDir = '/Users/guest1/Documents/work/options/'
inputFile = 'option-chain-ED-INFY-29-Jul-2021.csv'  #'option-chain-ED-NIFTY-04-Mar-2021.csv'
outputFile = 'INFY-option-chain-ED-INFY-29Jul-24Jun.csv'

df0 = pd.read_csv(workDir + inputFile)
df0.info(verbose=True)

filter = ['-']
df1 = df0[~df0['OI'].str.contains(filter[0])]

### INFY
nifty_close = 1544
lot = 600 #75
strike_increments = 20
iterations = 50
backwardation = 200 #go back how many points...
ATM_PUT_strike = int(math.ceil(nifty_close/20.0)) * 20
breakeven_filter = -5.0
away_from_maxP_filter = 10.1

### Nifty
"""
nifty_close = 15683
lot = 50 #75
strike_increments = 50
iterations = 30
backwardation = 200 #go back how many points...
ATM_PUT_strike = int(math.ceil(nifty_close/50.0)) * 50
breakeven_filter = -5.0
away_from_maxP_filter = 10.1
"""

"""
nifty_close = 34558
strike_increments = 100
lot = 25
iterations = 55
backwardation = 2000
breakeven_filter = -3.0
away_from_maxP_filter = 5.0
ATM_PUT_strike = int(math.ceil(nifty_close/100.0)) * 100
"""

# - BULL PUT SPREAD (Mildly Bullish)
## - Sell OTM PUT, and buy further OTM PUT
header = ['Sell Strike','Prem In','Sell IV', 'Sell OI','Buy Strike','Prem Out', 'Buy IV', 'Buy OI', \
          'Net Credit', 'Max Loss (ML)', 'Max Profit (MP)', 'BEP', 'CV(wip)', 'POP(wip)', 'R/R', \
          '% from ML', '% from BEP', '% from MP',  'P/L every % move', 'M ratio']

buy_PUT_Premium = 0.0
net_Premium_Paid = 0.0

with open(workDir + outputFile, 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',')
    
    #wr.writerow(["Last Close", str(nifty_close), "Filters ->","Risk > Reward","Positive BEP","High Outgo"])
    wr.writerow([g for g in header]) 
    
    #sell_PUT_Strike = ATM_PUT_strike - strike_increments  #replace with +500?
    
    sell_PUT_Strike = ATM_PUT_strike + backwardation
    print(sell_PUT_Strike)
    
    j = 0
    while j < iterations: 
        try:
            buy_PUT_Strike = sell_PUT_Strike - strike_increments

            df2 = df1[df1['STRIKE PRICE'].str.contains("{:,.2f}".format(sell_PUT_Strike))]
            df2 = df2.astype(str)

            #sell_PUT_Premium = float(df2["LTP.1"])
            #sell_PUT_Premium = float(str(df2["LTP.1"].values).replace("['","").replace("']","").replace(",",""))
            
            #sell_PUT_Premium = float(df2["LTP.1"])

            sell_PUT_Premium = 0.0

            if "," in str(df2["LTP.1"].values):
                sell_PUT_Premium = float(str(df2["LTP.1"].values).replace("['","").replace("']","").replace(",",""))
            else:
                sell_PUT_Premium = float(str(df2["LTP.1"].values).replace("['","").replace("']",""))
            
            sell_IV = float(df2["IV.1"])
            sell_Volume = str(df2["VOLUME.1"].values).replace("['","").replace("']","")
            sell_OI = str(df2["OI.1"].values).replace("['","").replace("']","")
            

            
        #iterate, with sell_PUT_Strike as OTM_PUT and decreasing OTM_PUT strikes
            i = 0
            while i < iterations:

                df3 = df1[df1['STRIKE PRICE'].str.contains("{:,.2f}".format(buy_PUT_Strike))]
                
                #buy_PUT_Premium = float(df3["LTP.1"])
                #buy_PUT_Premium = float(str(df3["LTP.1"].values).replace("['","").replace("']","").replace(",",""))
                if "," in str(df3["LTP.1"].values):
                    buy_PUT_Premium = float(str(df3["LTP.1"].values).replace("['","").replace("']","").replace(",",""))
                else:
                    if "['" in str(df3["LTP.1"].values):
                        buy_PUT_Premium = float(str(df3["LTP.1"].values).replace("['","").replace("']",""))
                    else:
                        buy_PUT_Premium = float(str(df3["LTP.1"].values).replace("[","").replace("]",""))
            
                buy_Volume = str(df3["VOLUME.1"].values).replace("['","").replace("']","")
                buy_OI = str(df3["OI.1"].values).replace("['","").replace("']","")
                net_Premium_Paid = float(sell_PUT_Premium) - float(buy_PUT_Premium)
                BEP = sell_PUT_Strike - net_Premium_Paid
                max_Profit = round(lot * net_Premium_Paid,2)
                max_Loss = round(-lot * (sell_PUT_Strike - buy_PUT_Strike - net_Premium_Paid),2)
                away_from_BEP = round(100*(BEP - nifty_close)/ nifty_close,2)
                net_Credit = round(lot * net_Premium_Paid)

                away_from_maxP = round(100*(sell_PUT_Strike - nifty_close)/ nifty_close,2)
                away_from_maxL = round(100*(nifty_close - buy_PUT_Strike)/ nifty_close,2)
  
                curr_value = 0
                if max_Profit > 0:    
                    if nifty_close > BEP:
                        curr_value = abs(nifty_close - BEP) * (max_Profit/(sell_PUT_Strike - BEP))
                    if nifty_close < BEP:
                        curr_value = abs(sell_PUT_Strike - nifty_close) * (max_Profit/(sell_PUT_Strike - BEP))
                if nifty_close > sell_PUT_Strike or curr_value > max_Profit:
                    curr_value = max_Profit
                    
                buy_IV = float(df3["IV.1"])

                filter = 0

                # POP = 100 - [(the credit received / strike price width) x 100]
                # POP calculation needs to be updated

                if filter == 1:
                    #if abs(max_Profit) > abs(max_Loss): #and away_from_BEP >=0.1:
                    if buy_PUT_Premium > 0.0 and sell_PUT_Premium > 0.0 and round(100*max_Profit/abs(max_Loss),2) > 50.0 \
                    and max_Profit > 3000.0 and abs(away_from_maxP) <= away_from_maxP_filter and buy_Volume != '0' \
                    and sell_Volume != '0' and away_from_BEP < 3.0 :
                        wr.writerow((sell_PUT_Strike, sell_PUT_Premium, sell_IV, sell_OI, buy_PUT_Strike, \
                                     buy_PUT_Premium, buy_IV, buy_OI, net_Credit, max_Loss, max_Profit, BEP \
                                     , round(curr_value,2) \
                                     , round(lot - net_Premium_Paid/(buy_PUT_Strike - sell_PUT_Strike),2) \
                                     , str(round(100*max_Profit/abs(max_Loss),2)) + '%', str(-away_from_maxL) + '%' 
                                     , str(away_from_BEP) + '%', str(away_from_maxP) + '%'
                                     , str(abs(round(max_Profit/away_from_maxP,2))) + \
                                     ' --> ' + str(abs(round(max_Loss/ away_from_maxL,2))) \
                                     , round(abs(round(max_Profit/away_from_maxP,2))/ abs(round(max_Loss/ away_from_maxL,2)),2) \
                            ))
                if filter == 0:
                    wr.writerow((sell_PUT_Strike, sell_PUT_Premium, sell_IV, sell_OI, buy_PUT_Strike, buy_PUT_Premium, \
                                 buy_IV, buy_OI , net_Credit, max_Loss, max_Profit, BEP \
                                 , round(curr_value,2), round(100 - net_Premium_Paid/(buy_PUT_Strike - sell_PUT_Strike),2) \
                                 , str(round(100*max_Profit/abs(max_Loss),2)) + '%', str(away_from_maxL) + '%' \
                                 , str(-away_from_maxP) + '%' , str(away_from_BEP) + '%'
                                 , str(abs(round(max_Profit/away_from_maxP,2))) + \
                                 ' --> ' + str(abs(round(max_Loss/ away_from_maxL,2))) \
                                 , round(abs(round(max_Profit/away_from_maxP,2))/ abs(round(max_Loss/ away_from_maxL,2)),2) \
                    ))
                        
                i += 1
                buy_PUT_Strike -= strike_increments
        except ValueError or TypeError or ZeroDivisionError as e:
            print('-->')
            print(sell_PUT_Strike)
            print(buy_PUT_Strike)
            print(sell_PUT_Premium)
            print(net_Premium_Paid)
            print(e)
            print('<--')
            
        sell_PUT_Strike -= strike_increments
        j += 1
            

