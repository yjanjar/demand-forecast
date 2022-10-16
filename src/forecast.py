import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

from plotly.subplots import make_subplots

import statsmodels.api as sm
from pylab import rcParams
import matplotlib.pyplot as plt

rcParams["figure.figsize"] = 15, 8
import itertools

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


class Forecast:
    """Class Forecast


    Attributes:

    """

    def __init__(
        self,
        costumer_ts: pd.Series,
        seasonality: int,
        val_start="2020-11-30",
        val_end="2021-10-31",
        test_start="2021-11-30",
        test_end="2022-06-30",
        exog=None,
        val_time=None,
        test_time=None,
    ):

        """Inits Forecast"""

        self.seasonality = seasonality
        self.costumer_ts = costumer_ts
        self.val_start, self.val_end = val_start, val_end
        self.test_start, self.test_end = test_start, test_end

    def get_decomp(self, ts, model="additive", period=12):
        """Get statistical decompositon of a time series. with fixed model and period

        Args:

           ts (pd.Series): time series


        Returns:
            None
        """

        decomposition = sm.tsa.seasonal_decompose(ts, model=model, period=period)
        fig = decomposition.plot()
        plt.show()

    def find_param(self, ts, max_p=2, trend="t", exog=None):

        """Get statistical decompositon of a time series. with fixed model and period

        Args:

           ts (pd.Series): time series


        Returns:
            None
        """

        p = d = q = range(0, max_p)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [
            (x[0], x[1], x[2], self.seasonality)
            for x in list(itertools.product(p, d, q))
        ]

        min = (None, None, float("inf"))
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                mod = sm.tsa.statespace.SARIMAX(
                    endog=ts,
                    exog=exog,
                    trend=trend,
                    order=param,
                    seasonal_order=param_seasonal,
                )

                results = mod.fit()
                if results.aic < min[2]:
                    min = (param, param_seasonal, results.aic)

                    print(
                        "ARIMA{}x{}12 - AIC:{}".format(
                            param, param_seasonal, results.aic
                        )
                    )

        return min[0], min[1]

    def get_predictions(
        self, ts, trend, param, s_param, validation=True, exog=None, exog_pred=None
    ):
        """Get Prediction based on endog and exog data.

        Args:

            ts (pd.Series): time series
            start_forecast: str
            end_forecast: str
            param: tuple(int) SARIMAX order,
            s_param: tuple(int) SARIMAX seasonal order,
            validation: bool Validation or Test data,
            exog: pd.DataFrame Exogenous data for the forecast


        Returns:
            None
        """
        if validation:

            start_forecast, end_forecast = self.val_start, self.val_end
        else:
            start_forecast, end_forecast = self.test_start, self.test_end

        if exog != None:
            mod = sm.tsa.statespace.SARIMAX(
                ts,
                exog=exog[: self.val_end],
                order=param,
                trend=trend,
                seasonal_order=s_param,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:

            mod = sm.tsa.statespace.SARIMAX(
                ts,
                exog=exog,
                order=param,
                trend=trend,
                seasonal_order=s_param,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

        results = mod.fit()

        pred = results.get_prediction(
            start=start_forecast, end=end_forecast, exog=exog_pred, dynamic=False
        )
        pred_ci = pred.conf_int()

        return pred, pred_ci

    def show_forecast(self, ts, pred_ts):

        """Evalute the prediction based on the know ground truth

        Args:

            ts (pd.Series): real time series
            pred_ts (pd.Series): predicted time series


        Returns:
            mae, rmse: (float, float)
        """

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name="Real Data")
        )
        fig.add_trace(
            go.Scatter(
                x=pred_ts.index,
                y=pred_ts.values,
                mode="lines+markers",
                name="Prediction",
            )
        )
        fig.add_vline(x="2018-06-30")
        fig.add_vline(x="2020-09-30")
        fig.add_vline(x="2021-10-31")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Value")
        fig.update_layout(title=ts.name + " Forecast")
        fig.show()

    def get_eval_metrics(self, ts, pred_ts):

        """Evalute the prediction based on the know ground truth

        Args:

            ts (pd.Series): real time series
            pred_ts (pd.Series): predicted time series


        Returns:
            mae, rmse: (float, float)
        """

        rmse = lambda act, pred: np.sqrt(mean_squared_error(act, pred))

        rmse_, mape = rmse(ts, pred_ts), mean_absolute_percentage_error(ts, pred_ts)
        print(f"RMSE: {rmse_}")
        print(f"MAPE: {mape}")
        return rmse, mape
