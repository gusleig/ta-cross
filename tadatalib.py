import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime, timedelta

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
# py.init_notebook_mode(connected=True)

import talib
import requests

altcoins = ['BTC_ETH', 'USDT_BTC']


def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''

    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas", start_date="2018-12-31", end_date="2019-01-29")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]

    return pd.DataFrame(series_dict)


def merge_btc(exchange_data):
    merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')


def get_btc_price():

    # exchanges = ['COINBASE', 'BITSTAMP', 'ITBIT', 'KRAKEN']
    exchanges = ['KRAKEN']

    exchange_data = {}

    for exchange in exchanges:
        exchange_code = 'BCHARTS/{}USD'.format(exchange)
        btc_exchange_df = get_quandl_data(exchange_code)
        exchange_data[exchange] = btc_exchange_df

    # btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Close')

    btc_usd_datasets = exchange_data['KRAKEN']

    # Remove "0" values
    btc_usd_datasets.replace(0, np.nan, inplace=True)

    return btc_usd_datasets


def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    # try:
    #    f = open(cache_path, 'rb')
    #    df = pickle.load(f)
    #    print('Loaded {} from cache'.format(json_url))
    # except (OSError, IOError) as e:
    #    print('Downloading {}'.format(json_url))
    #    # df = pd.read_json(json_url)
    #    df = pd.read_json(requests.get(json_url, verify=False).json())
    #    df.to_pickle(cache_path)
    #    print('Cached {} at {}'.format(json_url, cache_path))

    try:
        print('Downloading {}'.format(json_url))
        # df = pd.read_json(json_url)
        df = pd.DataFrame(requests.get(json_url, verify=False).json())
    except Exception as e:
        print(e)

    return df


def get_crypto_data(poloniex_pair, ndays=30):
    #  "period" (candlestick period in seconds; valid values are 300, 900, 1800, 7200, 14400, and 86400)
    base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'

    date_N_days_ago = datetime.now() - timedelta(days=ndays)

    # start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')  # get data from the start of 2015

    end_date = datetime.now()  # up until today

    period = 86400  # pull daily data (86,400 seconds per day)

    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, date_N_days_ago.timestamp(), end_date.timestamp(), period)

    data_df = get_json_data(json_url, poloniex_pair)

    data_df = data_df.set_index('date')

    return data_df


def get_altcoin():
    # markets = ["BTCUSDT", "ETHBTC", "XLMBTC", "ADABTC", "VETBTC"]

    altcoins = ['BTC_ETH', 'USDT_BTC']

    altcoin_data = {}
    for altcoin in altcoins:
        #coinpair = 'BTC_{}'.format(altcoin)
        coinpair = altcoin
        crypto_price_df = get_crypto_data(coinpair)
        altcoin_data[altcoin] = crypto_price_df

    return altcoin_data


def get_altcoin_usd(altcoin_data, btc_usd_datasets):

    # Calculate USD Price as a new column in each altcoin dataframe
    for altcoin in altcoin_data.keys():
        altcoin_data[altcoin]['price_usd'] = altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']


def getRSI(closePrices, lowerlimit=40, higherlimit=75):
    buySignal = False
    sellSignal = False

    rsi = talib.RSI(closePrices, timeperiod=14)

    if rsi[-1] < lowerlimit:
        buySignal = True
    if rsi[-1] > higherlimit:
        sellSignal = True

    return round(rsi[-1], 2), buySignal, sellSignal


def getBB(closePrices):
    upper, middle, lower = talib.BBANDS(closePrices, 20, 2, 2)
    return upper[-1], middle[-1], lower[-1]


def getMACD(closePrices):
    macd, macdsignal, macdhist = talib.MACD(closePrices, fastperiod=12, slowperiod=26, signalperiod=9)
    return macd, macdsignal, macdhist


def getMA(closePrices, period):
    real = talib.MA(closePrices, timeperiod=period, matype=0)
    return real


def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Generate a scatter plot of the entire dataframe'''
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))

    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels=not seperate_y_axis,
            type=scale
        )
    )

    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale)

    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'

    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index,
            y=series,
            name=label_arr[index],
            visible=visibility
        )

        # Add seperate axis for the series
        if seperate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config
        trace_arr.append(trace)

    fig = go.Figure(data=trace_arr, layout=layout)
    py.iplot(fig)


def main():

    a = get_altcoin()

    combined_df = merge_dfs_on_column(list(a.values()), list(a.keys()), 'close')

    alt_close = []

    f = get_btc_price()

    for alt in a:

        alt_close.append(np.array(a[alt]['close'][1:]))

    close = np.array(f['Close'][1:])

    modclose = np.zeros(len(close))

    for i in range(len(close)):
        modclose[i] = float(close[i])

    sma = talib.SMA(modclose, timeperiod=30)

    rsi = talib.RSI(modclose, timeperiod=14)

    btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')

    btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken['Weighted Price'])

    py.iplot([btc_trace])

    df_scatter(f, 'Bitcoin Price (USD) By Exchange')


if __name__ == '__main__':
    main()