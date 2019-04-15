import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
from pandas.plotting import register_matplotlib_converters
import time
from pycse import regress
import uncertainties as u
from uncertainties import unumpy
from scipy.optimize import fsolve
from matplotlib.dates import DateFormatter
from matplotlib import rcParams

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
    b1 = u.ufloat((p1[0], se1[0]))
    m1 = u.ufloat((p1[1], se1[1]))

    b2 = u.ufloat((p2[0], se2[0]))
    m2 = u.ufloat((p2[1], se2[1]))

    T_intersection = (b2 - b1) / (m1 - m2)
    # print(T_intersection)

    T_intersect_nv = T_intersection.nominal_value

    return datetime.datetime.fromtimestamp(int(T_intersect_nv))


def cross(data1, data, prices_a, prices_b):
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
    b1 = u.ufloat((p1[0], se1[0]))
    m1 = u.ufloat((p1[1], se1[1]))

    b2 = u.ufloat((p2[0], se2[0]))
    m2 = u.ufloat((p2[1], se2[1]))

    T_intersection = (b2 - b1) / (m1 - m2)
    # print(T_intersection)

    T_intersect_nv = T_intersection.nominal_value

    print(datetime.datetime.fromtimestamp(int(T_intersect_nv)))

    fig, ax = plt.subplots()

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

    x = datetime.datetime.fromtimestamp(int((T_intersect_nv - 2 * T_intersection.std_dev())))

    y = datetime.datetime.fromtimestamp(int((T_intersect_nv + 2 * T_intersection.std_dev())))

    ax.plot([x,
             y],
            [(b1 + m1 * T_intersection).nominal_value,
             (b1 + m1 * T_intersection).nominal_value],
            'g-', lw=3, label='$\pm 2 \sigma$')

    ax.legend(loc='best')
    plt.show()


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


def main():
    # %config InlineBackend.figure_format = 'retina'
    # %matplotlib inline

    btc = web.get_data_yahoo('BTC-USD', start=datetime.datetime(2017, 1, 1), end=date.today())

    btc_adj = btc['Adj Close']

    # btc_adj.plot(lw=2.5, figsize=(12, 5))
    # plt.show()

    short_window = 50
    mid_window = 200

    signals = pd.DataFrame(index=btc_adj.index)
    signals['signal'] = 0.0

    roll_d10 = btc_adj.rolling(window=short_window).mean()
    roll_d50 = btc_adj.rolling(window=mid_window).mean()

    signals['short_mavg'] = roll_d10
    signals['mid_mavg'] = roll_d50

    d10_vector = []

    last_date = int(time.mktime(roll_d10[-1:].index[0].timetuple()))
    # 20_date = int(time.mktime(roll_d10[-20:].index[0].timetuple()) )

    # data_normal = [roll_d10[-1:].index[0], roll_d10[-20:].index[0]]
    data_normal = roll_d10[-20:].index.tolist()

    # data_ts = [int(time.mktime(roll_d10[-1:].index[0].timetuple()) ), int(time.mktime(roll_d10[-20:].index[0].timetuple()) )]
    data_ts = (roll_d10[-20:].index.astype(np.int64) // 10 ** 9).tolist()

    prices_a = roll_d10[-20:].tolist()
    prices_b = roll_d50[-20:].tolist()

    # cross(data_normal, data_ts, prices_a, prices_b)

    # plot_crosses(signals, btc_adj, roll_d10, roll_d50)
    print(cross_date(data_normal, data_ts, prices_a, prices_b))


if __name__ == '__main__':
    main()