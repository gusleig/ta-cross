import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import datetime
from datetime import date
from pandas.plotting import register_matplotlib_converters
from pycse import regress
import uncertainties as u
# from uncertainties import unumpy
# from scipy.optimize import fsolve
from matplotlib.dates import DateFormatter
# import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib import rcParams
# from scipy.signal import find_peaks
import talib

# from peakdetect import peakdetect

rcParams.update({'figure.autolayout': True})

plt.style.use('fivethirtyeight')

register_matplotlib_converters()


def cross_date(data1, data, prices_a, prices_b):
    D = np.array(data1)
    T = np.array(data)
    E1 = np.array(prices_a)
    E2 = np.array(prices_b)

    A = np.column_stack([T ** 0, T])

    p1, pint1, se1 = regress(A, E1, alpha=0.05)

    p2, pint2, se2 = regress(A, E2, alpha=0.05)

    # Now we have two lines: y1 = m1*T + b1 and y2 = m2*T + b2
    # they intersect at m1*T + b1 = m2*T + b2
    # or at T = (b2 - b1) / (m1 - m2)
    b1 = u.ufloat(p1[0], se1[0])
    m1 = u.ufloat(p1[1], se1[1])

    b2 = u.ufloat(p2[0], se2[0])
    m2 = u.ufloat(p2[1], se2[1])

    T_intersection = (b2 - b1) / (m1 - m2)
    # print(T_intersection)

    T_intersect_nv = T_intersection.nominal_value

    return datetime.datetime.fromtimestamp(int(T_intersect_nv))


def plot_crosses(signals, btc_adj, roll_d10, roll_d50, short_window):
    signals['signal'][short_window:] = np.where(
        signals['short_mavg'][short_window:] > signals['mid_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    plt.figure(figsize=(14, 7))
    plt.plot(btc_adj.index, btc_adj, lw=3, alpha=0.8, label='Original observations')
    plt.plot(btc_adj.index, roll_d10, lw=3, alpha=0.8, label='Rolling mean (window 50)')
    plt.plot(btc_adj.index, roll_d50, lw=3, alpha=0.8, label='Rolling mean (window 200)')
    plt.plot(signals.loc[signals.positions == 1.0].index,
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='r', label='buy')

    plt.plot(signals.loc[signals.positions == -1.0].index,
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k', label='sell')

    plt.title('BTC-USD Adj Close Price (The Technical Approach)')
    plt.tick_params(labelsize=12)
    plt.legend(loc='upper left', fontsize=12)

    plt.show()

    # plt.savefig('images/intersection-1.png')


def rsi_peaks(closePrices, candles=10):

    np.random.seed(42)

    fig, ax = plt.subplots(facecolor='#07000d')

    ax.set_facecolor('#07000d')

    #fig = plt.figure(facecolor='#07000d')
    # ax = plt.subplot2grid((6,4), (0,0), axisbg='#07000d')

    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'

    rsi = talib.RSI(closePrices, timeperiod=14)

    # https://gist.github.com/sixtenbe/1178136#file-peakdetect-py
    # cb = np.array([-0.010223, ...])
    # peaks = peakdetect(cb, lookahead=100)

    minima = []
    maxima = []

    rsi_n = rsi[-candles:]

    for i in range(1, candles-1):

        if rsi_n[i] < rsi_n[i-1] and rsi_n[i] < rsi_n[i+1]:
            # local minima
            minima.append(i)
        if rsi_n[i] > rsi_n[i - 1] and rsi_n[i] > rsi_n[i + 1]:
            # local maxima
            maxima.append(i)

    # peaks, _ = find_peaks(rsi_n.values, height=60)

    ax.plot(rsi[-candles:].index, rsi[-candles:].values, rsiCol, label="RSI", linewidth=1.5)
    # plt.plot(rsi_n[peaks].index, rsi_n[peaks].values, "x")

    ax.fill_between(rsi[-candles:].index, rsi[-candles:].values, 70, where=(rsi[-candles:].values >= 70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    ax.fill_between(rsi[-candles:].index, rsi[-candles:].values, 30, where=(rsi[-candles:].values <= 30), facecolor=posCol, edgecolor=posCol, alpha=0.5)

    ax.set_yticks([30, 70])
    ax.yaxis.label.set_color("w")
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.tick_params(axis='y', colors='w')
    ax.tick_params(axis='x', colors='w')
    plt.ylabel('RSI')

    ax.plot(rsi_n[maxima], "x")

    fig.autofmt_xdate()

    ax.axhline(70, color=negCol)
    ax.axhline(30, color=posCol)

    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.ylabel('RSI')

    ax.grid(True, color='w')

    plt.show()


def plot_next_cross(data1, data, prices_a, prices_b):
    D = np.array(data1)
    T = np.array(data)
    E1 = np.array(prices_a)
    E2 = np.array(prices_b)

    A = np.column_stack([T ** 0, T])

    p1, pint1, se1 = regress(A, E1, alpha=0.05)

    p2, pint2, se2 = regress(A, E2, alpha=0.05)

    # Now we have two lines: y1 = m1*T + b1 and y2 = m2*T + b2
    # they intersect at m1*T + b1 = m2*T + b2
    # or at T = (b2 - b1) / (m1 - m2)
    b1 = u.ufloat(p1[0], se1[0])
    m1 = u.ufloat(p1[1], se1[1])

    b2 = u.ufloat(p2[0], se2[0])
    m2 = u.ufloat(p2[1], se2[1])

    T_intersection = (b2 - b1) / (m1 - m2)
    # print(T_intersection)

    T_intersect_nv = T_intersection.nominal_value

    print(datetime.datetime.fromtimestamp(int(T_intersect_nv)))

    fig, ax = plt.subplots()

    # plt.figure()

    ax.set(xlabel="Date", ylabel="Price (USD)", )
    ax.set_title("Golden/Death Cross\nEstim. Date")

    myFmt = DateFormatter("%m-%d")

    ax.xaxis.set_major_formatter(myFmt)
    ax.tick_params(axis='x', rotation=45)

    # plot the data, the fits and the intersection and \pm 2 \sigma.
    ax.plot(D, E1, 'bo ', label='MA50')
    ax.plot(D, np.dot(A, p1), 'b-')
    ax.plot(D, E2, 'ro ', label='MA200')
    ax.plot(D, np.dot(A, p2), 'r-')

    a = datetime.datetime.fromtimestamp(int(T_intersect_nv))
    b = (b1 + m1 * T_intersection).nominal_value

    ax.plot(a, b, 'go', ms=13, alpha=0.2, label='Intersection')

    label = "{:.2f}".format(b) + " - " + str(a.date())

    ax.annotate(label, xy=(a, b), xycoords='data', xytext=(a, b),
                arrowprops=dict(facecolor='black', shrink=0.05))

    x = datetime.datetime.fromtimestamp(int((T_intersect_nv - 2 * T_intersection.std_dev)))

    y = datetime.datetime.fromtimestamp(int((T_intersect_nv + 2 * T_intersection.std_dev)))

    ax.plot([x,
             y],
            [(b1 + m1 * T_intersection).nominal_value,
             (b1 + m1 * T_intersection).nominal_value],
            'g-', lw=3, label='$\pm 2 \sigma$')

    ax.legend(loc='best')
    plt.show(block=True)


def main():

    # number of daily candles to analyze

    x = 60

    date_str = (datetime.datetime.now() - datetime.timedelta(days=x)).date()

    btc = web.get_data_yahoo('BTC-USD', start=datetime.datetime(2017, 1, 1), end=date.today())

    btc_adj = btc['Adj Close']

    # show rsi peaks (beta)
    # rsi_peaks(btc_adj)

    # btc_adj.plot(lw=2.5, figsize=(12, 5))
    # plt.show()

    short_window = 150
    mid_window = 600

    signals = pd.DataFrame(index=btc_adj.index)
    signals['signal'] = 0.0

    roll_d10 = btc_adj.rolling(window=short_window).mean()
    roll_d50 = btc_adj.rolling(window=mid_window).mean()

    signals['short_mavg'] = roll_d10
    signals['mid_mavg'] = roll_d50


    data_normal = roll_d10[-x:].index.tolist()

    # data_ts = [int(time.mktime(roll_d10[-1:].index[0].timetuple()) ), int(time.mktime(roll_d10[-20:].index[0].timetuple()) )]
    # convert time to timestamp to act as a number
    data_ts = (roll_d10[-x:].index.astype(np.int64) // 10 ** 9).tolist()

    prices_a = roll_d10[-x:].tolist()
    prices_b = roll_d50[-x:].tolist()

    plot_next_cross(data_normal, data_ts, prices_a, prices_b)

    # plot_crosses(signals, btc_adj, roll_d10, roll_d50)

    print(cross_date(data_normal, data_ts, prices_a, prices_b))


if __name__ == '__main__':
    main()