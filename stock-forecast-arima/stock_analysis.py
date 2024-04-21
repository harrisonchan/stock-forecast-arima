import yfinance
import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import numpy as np
import datetime


def get_data(ticker, start, end):
    res = yfinance.download(ticker, start, end)
    if len(res) == 0:
        print(f"Could not find data for {ticker}")
        return
    return pd.DataFrame({ticker: res["Close"]})


def difference(data, interval=1):
    diff = list()
    for i in range(len(data)):
        if i < interval:
            diff.append(np.nan)
        else:
            val = data[i] - data[i - interval]
            diff.append(val)
    return pd.Series(diff, index=data.index).dropna()


def plot(data, lags=None, title="", figsize=(14, 8)):
    fig = plt.figure()
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    data.plot(ax=ts_ax)
    ts_ax.set_title(title)
    data.plot(ax=hist_ax, king="hist", bins=25)
    hist_ax.set_title("Histogram")
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax


def test_adf(series):
    adf_result = st.adfuller(series, store=True)
    print("ADF Test Results: ")
    print("Test Statistic: %.4f" % adf_result[0])
    print("p-value: %.10f" % adf_result[1])
    print("Critical Values: ")
    for key, value in adf_result[2].items():
        print("\t%s: %0.3f" % (key, value))
    return adf_result


# Just trying not to get random-walk or 0,0,0 here.
# Needs to be much improved
def get_order(data, interval, tries, percentage=0.9, order=(0, 0, 0)):
    if tries <= 0:
        print("out of tries, leaving...")
        return [difference(data, interval), order]
    data_diff = difference(data, interval)
    adf_result = test_adf(data_diff)
    test_statistic = adf_result[0]
    critical_values = adf_result[2]
    for key, value in critical_values.items():
        if test_statistic > value:
            return get_order(data, interval + 1, tries - 1, percentage, order)
        split = int(len(data_diff) * percentage)
        training = data_diff[0:split]
        testing = data_diff[split:]
        stepwise_fit = auto_arima(
            training, trace=True, error_action="ignore", suppress_warnings=True
        )
        stepwise_fit.fit(training)
        print(stepwise_fit.summary())
        new_order = stepwise_fit.get_params().get("order")
        print("new_order", new_order)
        if (
            new_order == (0, 1, 0)
            or new_order == (0, 0, 0)
            or new_order == (0, 0, 1)
            or new_order == (5, 0, 0)
            or new_order == (5, 0, 1)
        ):
            print("Going to try and find a better fit...")
            return get_order(data, interval + 1, tries - 1, percentage, new_order)
        else:
            return [data_diff, new_order]


def use_arima(data, order, percentage):
    model = ARIMA(data, order=order)
    model = model.fit()
    print(model.summary())
    split = int(len(data) * percentage)
    training = data.iloc[0:split]
    testing = data.iloc[split:]
    start = len(training)
    end = len(training) + len(testing) - 1
    prediction = model.predict(start=start, end=end, type="levels").rename(
        "Predictions"
    )
    # prediction.plot(legend=True)
    # data.plot(legend=True)
    # plt.show()
    return [model, prediction]


def make_future_predictions(model, data, order, from_date, days_to_predict=60):
    full_model = ARIMA(data, order=order)
    full_model = full_model.fit()
    print(data.tail())
    from_date_dt = datetime.datetime.strptime(from_date, "%Y-%m-%d")
    end_date = from_date_dt + datetime.timedelta(days=days_to_predict)
    end_date_str = end_date.strftime("%Y-%m-%d")
    dates = pd.date_range(start=from_date, end=end_date_str)
    prediction = full_model.predict(
        start=len(data), end=len(data) + days_to_predict, type="levels"
    ).rename("Future Predictions")
    prediction.index = dates
    print(prediction)
    return [full_model, prediction]


def main():
    TICKER = "AAPL"
    START = "2018-01-01"
    END = "2024-01-01"
    data = get_data(TICKER, START, END)
    test_adf(data)
    PERCENTAGE = 0.9
    INTERVAL_START = 1
    TRIES = 5
    INITIAL_ORDER = (2, 0, 2)
    data_diff, order = get_order(
        data[TICKER], INTERVAL_START, TRIES, PERCENTAGE, INITIAL_ORDER
    )
    [model, prediction] = use_arima(data_diff, order, PERCENTAGE)
    [full_model, future_prediction] = make_future_predictions(
        model, data_diff, order, END, 60
    )
    prediction.plot(legend=True)
    data_diff.plot(legend=True)
    future_prediction.plot(figsize=(12, 5), legend=True)
    plt.show()


main()
