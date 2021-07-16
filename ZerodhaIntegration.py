# -*- coding: utf-8 -*-

import requests
from kiteconnect import KiteConnect
from KeyInfo import _Zerodha_Request_Token, _API_Secret, _Zerodha_Access_Token, _API_Key
from TAUtils import saveToRedis, getFromRedis
import datetime
import logging
import json
import pandas as pd

#redis_conn = redis.Redis(host='localhost', port=6379, db=0)

API_Calls_Dir = "/Users/guest1/Downloads/bhavcopyDownloads/API_Calls/"
today = datetime.date.today()

logging.basicConfig(filename = API_Calls_Dir + 'Zerodha_' + today.strftime('%d%h%Y') + '.log', level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

kite = KiteConnect(api_key = _API_Key)

df_trades = pd.DataFrame()

def callZerodhaAPI(tran_type, stock, **kwargs):
    """
    *kwargs: Quantity, Variety, Exchange, Order Type, Product, Price, Validity.....
    """

    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 

    # BUY, SQLL, FETCH, CANCEL, CONVERT
    if tran_type == 'BUY':
        logging.info(' Zerodha Buy ')
        #placeBuyOrder()
    if tran_type == 'SELL':
        logging.info(' Zerodha Sell ')
        #placeSellOrder()
    if tran_type == 'FETCH':
        logging.info(' Zerodha Fetch ')
        #getBulkQuote()
    if tran_type == 'CANCEL':
        logging.info(' Zerodha Cancel Open Order ')
        #cancelOrder()

def placeBuyOrder(order_stock, order_quantity, order_price, order_squareoff, order_stoploss, order_trailing_stoploss):

    logging.info(' Inside placeBuyOrder --> Entry')
    try:
        order_id = kite.place_order(variety    = kite.VARIETY_BO  # or VARIETY_REGULAR
                                ,tradingsymbol  = order_stock
                                ,exchange       = kite.EXCHANGE_NSE
                                ,transaction_type= kite.TRANSACTION_TYPE_BUY
                                ,quantity       = order_quantity
                                ,order_type     = kite.ORDER_TYPE_LIMIT
                                ,product        = kite.PRODUCT_MIS
                                ,price          = order_price
                                ,squareoff      = order_squareoff
                                ,stoploss       = order_stoploss
                                ,trailing_stoploss= order_trailing_stoploss
                                ,validity       = kite.VALIDITY_DAY)

        logging.info("Order placed. ID is: {}".format(order_id))
    except Exception as e:
        logging.info("Order placement failed: {}".format(e.message))
    
    # TO DO: Update trades file with order-id, order-status
    logging.info('Inside placeBuyOrder --> Exit')


def getBulkQuote(instruments_list):
    logging.info('Inside getBulkQuote --> Entry')

    try:
        quote_json = kite.quote(instruments_list)
    except Exception as e:
        logging.info(" Bulk Quote failed : {}".format(e.message))

    # TO DO: get list of instruments from input file, check their quotes, and update 'Morning' file --> 9:20 am call
    logging.info('Inside getBulkQuote --> Exit')


def placeSellOrder(order_id, order_stock, order_quantity, order_price, order_squareoff, order_stoploss, order_trailing_stoploss):
    logging.info('Inside placeSellOrder --> Entry')

    try:
        order_id = kite.place_order(variety    = kite.VARIETY_REGULAR 
                                ,tradingsymbol  = order_stock
                                ,exchange       = kite.EXCHANGE_NSE
                                ,transaction_type= kite.TRANSACTION_TYPE_SELL
                                ,quantity       = order_quantity
                                ,order_type     = kite.ORDER_TYPE_LIMIT
                                ,product        = kite.PRODUCT_MIS
                                ,price          = order_price
                                ,squareoff      = order_squareoff
                                ,stoploss       = order_stoploss
                                ,trailing_stoploss= order_trailing_stoploss
                                ,validity       = kite.VALIDITY_DAY)

        logging.info("Order placed. ID is: {}".format(order_id))
    except Exception as e:
        logging.info("Order placement failed: {}".format(e.message))

    # TO DO: Update trades file with order-id, order-status
    logging.info('Inside placeSellOrder --> Exit')

#def cancelOpenOrder():
    # check all orders placed, the status of order execution, and cancel any orders still open


"""
https://kite.trade/docs/pykiteconnect/v3/#kiteconnect.KiteConnect.quote

def login_url(self)
def __init__(self, api_key, access_token=None, root=None, debug=False, timeout=None, proxies=None, pool=None, disable_ssl=False)

def generate_session(self, request_token, api_secret)
def set_access_token(self, access_token)
def renew_access_token(self, refresh_token, api_secret)

def quote(self, *instruments)
def ltp(self, *instruments)

def place_order(self, variety, exchange, tradingsymbol, transaction_type, quantity, product, order_type, price=None, validity=None, disclosed_quantity=None, trigger_price=None, squareoff=None, stoploss=None, trailing_stoploss=None, tag=None)

def trades(self)

def modify_order(self, variety, order_id, parent_order_id=None, quantity=None, price=None, order_type=None, trigger_price=None, validity=None, disclosed_quantity=None)

def positions(self)

def trades(	self)
Retrieve the list of trades executed (all or ones under a particular order).
An order can be executed in tranches based on market conditions. These trades are individually recorded under an order.
order_id is the ID of the order (optional) whose trades are to be retrieved. If no order_id is specified, all trades for the day are returned.

"""

"""
_Zerodha_Login_Url = 'https://kite.zerodha.com/connect/login?v=3&api_key=xxx'
_Zerodha_API_endpoint = 'https://api.kite.trade'
_Zerodha_Request_Token = ''
#_API_Secret = ''
_Zerodha_Access_Token = ''
#_API_Key = ''

# Redirect the user to the login url obtained
# from kite.login_url(), and receive the request_token
# from the registered redirect url after the login flow.
# Once you have the request_token, obtain the access_token
# as follows.

kite = KiteConnect(api_key = _API_Key)

data = kite.generate_session(_Zerodha_Request_Token, api_secret = _API_Secret)
kite.set_access_token(data["access_token"])
"""