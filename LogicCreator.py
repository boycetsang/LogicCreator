import time
from collections import OrderedDict
from Logic import *
from StrategySettings import StrategySettings
from StrategyLogic import StrategyLogic
from StrategyData import StrategyData
from StrategyCalculator import StrategyCalculator
from StrategyOutput import StrategyOutput
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

pd.set_option("mode.chained_assignment", None)


class Strategy:
    """Base object for Strategy to contain all the relevant variables

    Parameters:
    settings(dict): contains file/locale settings such as timezone, file paths
    dataSettings{dict}: contains date-related settings
    logicSettings(dict): contains logic-related settings, such as if a logic module is enabled, and its parameters
    calSettings{dict}: contains PnL-calculation settings, such as slippery
    outputSettings(dict): contains file and graphical output settings

   """

    def __init__(
        self,
        settings={},
        dataSettings={},
        logicSettings={},
        calSettings={},
        outputSettings={},
    ):
        self.strategySettings = StrategySettings(settings)
        self.strategyLogic = StrategyLogic(logicSettings)
        self.strategyData = StrategyData(
            dataSettings, dataNeededDaysBefore=self.strategyLogic.dataNeededDaysBefore
        )
        self.strategyCalculator = StrategyCalculator(calSettings)
        self.strategyOutput = StrategyOutput(outputSettings)

    def __str__(self):
        return "\n".join(
            [
                str(self.strategySettings),
                str(self.strategyData),
                str(self.strategyLogic),
                str(self.strategyCalculator),
                str(self.strategyOutput),
            ]
        )

    def dataClean(self, date, timeSeries):
        """Data cleanup step to trim the time series, reset in defined time zone, and prepare pandas DataFrame with frequency

        Parameters:
        date(str): date to extract time series in format "YYYYMMDD" 
        timeSeries(pd.Series): time series to be processed

        Return:
        timeSeries(pd.Series): processed time series
       """
        timeSeries["datetime"] = pd.to_datetime(timeSeries.tradeTime, utc=True)
        timeSeries = timeSeries.set_index("datetime")
        if not isinstance(
            timeSeries.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        ):
            timeSeries.index = timeSeries.index.tz_localize(
                self.strategySettings.settings["timezoneInRecord"]
            )
        else:
            timeSeries.index = timeSeries.index.tz_convert(
                self.strategySettings.settings["timezoneInRecord"]
            )
        timeSeries = timeSeries.price

        timeSeries.name = "price"
        self.indexStartTime = (
            pd.to_datetime(
                str(date) + " " + self.strategySettings.settings["indexStartTime"],
                format="%Y%m%d %H:%M:%S.%f",
            )
            .tz_localize(self.strategySettings.settings["timezoneInIndex"])
            .tz_convert(self.strategySettings.settings["timezoneInRecord"])
        )
        self.indexEndTime = (
            pd.to_datetime(
                str(date) + " " + self.strategySettings.settings["indexEndTime"],
                format="%Y%m%d %H:%M:%S.%f",
            )
            .tz_localize(self.strategySettings.settings["timezoneInIndex"])
            .tz_convert(self.strategySettings.settings["timezoneInRecord"])
        )

        self.tradeStartTime = (
            pd.to_datetime(
                str(date) + " " + self.strategySettings.settings["tradeStartTime"],
                format="%Y%m%d %H:%M:%S.%f",
            )
            .tz_localize(self.strategySettings.settings["timezoneInIndex"])
            .tz_convert(self.strategySettings.settings["timezoneInRecord"])
        )
        self.tradeEndTime = (
            pd.to_datetime(
                str(date) + " " + self.strategySettings.settings["tradeEndTime"],
                format="%Y%m%d %H:%M:%S.%f",
            )
            .tz_localize(self.strategySettings.settings["timezoneInIndex"])
            .tz_convert(self.strategySettings.settings["timezoneInRecord"])
        )

        self.lunchStartTime = (
            pd.to_datetime(
                str(date) + " " + self.strategySettings.settings["lunchStartTime"],
                format="%Y%m%d %H:%M:%S.%f",
            )
            .tz_localize(self.strategySettings.settings["timezoneInIndex"])
            .tz_convert(self.strategySettings.settings["timezoneInRecord"])
        )
        self.lunchEndTime = (
            pd.to_datetime(
                str(date) + " " + self.strategySettings.settings["lunchEndTime"],
                format="%Y%m%d %H:%M:%S.%f",
            )
            .tz_localize(self.strategySettings.settings["timezoneInIndex"])
            .tz_convert(self.strategySettings.settings["timezoneInRecord"])
        )
        grouped = timeSeries.groupby(level=0)
        timeSeries = grouped.last()
        timeSeries = timeSeries[
            (timeSeries.index > self.indexStartTime)
            & (timeSeries.index < self.indexEndTime)
            & (
                (timeSeries.index > self.lunchEndTime)
                | (timeSeries.index < self.lunchStartTime)
            )
        ]
        timeSeries = timeSeries.astype(float)
        # treating the insufficient data case
        # if len(timeSeries) != 0:
        #     timeSeries[self.indexEndTime] = timeSeries[
        #         (timeSeries.index <= self.indexEndTime)
        #     ].iloc[-1]
        return timeSeries

    def evalStratDay(self, date, data=None, positionClass=None):
        """Worker function to execute logic on the time series

        Parameters:
        date(pd.Datetime): date to execute strategy 
        data(pd.Series): use this time series instead of the one stored in the object
        positionClass(position): Use this position class instead of the default one
       """
        date = date.date()
        if data is None:
            if not date in self.strategyData.dateFileDict.keys():
                return
            self.strategyData.currentDate = date
            self.strategyData.timeSeries = pd.read_csv(
                self.strategyData.dateFileDict[date], na_values=["NA"]
            )
        else:
            self.strategyData.currentDate = date
            self.strategyData.timeSeries = data

        self.strategyData.timeSeries = self.dataClean(
            date, self.strategyData.timeSeries
        )
        self.strategyData.timeSeriesTick = self.strategyData.timeSeries
        self.strategyData.timeSeriesTrade = self.strategyData.timeSeries.resample(
            str(int(float(self.strategySettings.settings["interval"]) * 60)) + "S"
        ).first()
        if self.strategyData.timeSeriesTrade.empty:
            return
        self.strategyLogic.performComputeForDay(
            self, self.strategyData.timeSeriesTick, self.strategyData.timeSeriesTrade
        )
        if positionClass is not None:
            val = self.strategyData.timeSeriesTrade.iloc[-1]
            ind = self.strategyData.timeSeriesTrade.index[-1]
            self.strategyLogic.performComputeForAction(self, ind, val)
            if np.isnan(val):
                return
            for position in self.strategyCalculator.openPositionList:
                if position.type == "buy" and (
                    self.strategyLogic.getBuyExitCondition(self, position, ind)
                ):
                    position.close(ind, val)
                    self.strategyLogic.getExitReport(self, position)
                if position.type == "sell" and (
                    self.strategyLogic.getSellExitCondition(self, position, ind)
                ):
                    position.close(ind, val)
                    self.strategyLogic.getExitReport(self, position)
            # if ind in [p.openTime for p in self.strategyCalculator.openPositionList]:
            #     return
            if self.strategyLogic.getBuyEntranceCondition(self, ind):
                position = positionClass(self, ind, val, "buy")
                position = self.strategyLogic.getEntranceReport(self, position)
            if self.strategyLogic.getSellEntranceCondition(self, ind):
                position = positionClass(self, ind, val, "sell")
                position = self.strategyLogic.getEntranceReport(self, position)
            return
        for ind, val in self.strategyData.timeSeriesTrade.iteritems():
            self.strategyLogic.performComputeForAction(self, ind, val)
            if np.isnan(val):
                continue
            for position in self.strategyCalculator.openPositionList:
                if position.type == "buy" and (
                    self.strategyLogic.getBuyExitCondition(self, position, ind)
                ):
                    position.close(ind, val)
                    self.strategyLogic.getExitReport(self, position)
                if position.type == "sell" and (
                    self.strategyLogic.getSellExitCondition(self, position, ind)
                ):
                    position.close(ind, val)
                    self.strategyLogic.getExitReport(self, position)
            if self.strategyLogic.getBuyEntranceCondition(self, ind):
                position = self.strategyCalculator.Position(self, ind, val, "buy")
                position = self.strategyLogic.getEntranceReport(self, position)
            if self.strategyLogic.getSellEntranceCondition(self, ind):
                position = self.strategyCalculator.Position(self, ind, val, "sell")
                position = self.strategyLogic.getEntranceReport(self, position)
        realPnL = self.strategyOutput.getRealPnL(self)
        if realPnL is None:
            realPnL = 0

        if not self.strategyCalculator.positionList:
            self.strategyCalculator.positionList = None
        import copy

        if (self.strategyCalculator.positionList) is None:
            self.strategyCalculator.positionList = []
            return []
        result = list(self.strategyCalculator.positionList)
        self.strategyCalculator.positionList = []
        return result

    def evalStrat(self, silence=False):
        # evaluate strategy using 1 thread. This function does not use joblib.
        date_cnt = 0
        date_total = len(
            set(self.strategyData.tradeDateRange.date).intersection(
                set(self.strategyData.dateFileDict.keys())
            )
        )
        if not silence:
            print("Backtesting in progress ...")
        for date in self.strategyData.tradeDateRange:
            date_key = date.date()
            if not date_key in self.strategyData.dateFileDict.keys():
                continue
            result = self.evalStratDay(date)
            self.strategyCalculator.positionList += result

            date_cnt += 1
            if not silence:
                self.strategyOutput.updateProgress(date_cnt / date_total)

    def evalStratParallel(self, silence=False, num_core=0):
        # evaluate strategy using parallel process
        from joblib import Parallel, delayed

        import multiprocessing

        date_cnt = 0
        date_total = len(
            set(self.strategyData.tradeDateRange.date).intersection(
                set(self.strategyData.dateFileDict.keys())
            )
        )
        if not silence:
            print("Backtesting in progress ...")
        if num_core == 0:
            num_cores = max(multiprocessing.cpu_count() - 1, 1)

        results = Parallel(n_jobs=num_cores, verbose=11)(
            delayed(self.evalStratDay)(date)
            for date in self.strategyData.tradeDateRange
        )
        results = [result for result in results if result is not None]

        results = [item for sublist in results for item in sublist]

        self.strategyCalculator.positionList = results

    def outputGraphsParallel(self, silence=False, num_core=0):
        # produce graph using parallel process
        from joblib import Parallel, delayed
        import multiprocessing

        if not silence:
            print("Graph production in progress ...")
        if num_core == 0:
            num_cores = max(multiprocessing.cpu_count() - 1, 1)

        Parallel(n_jobs=num_cores, verbose=51)(
            delayed(self.strategyOutput.outputGraphsDay)(self, date)
            for date in self.strategyData.tradeDateRange
        )

    def __add__(self, other):
        self.strategyCalculator.positionList = (
            self.strategyCalculator.positionList + other.strategyCalculator.positionList
        )
        return self


def defaultSettings(settings):
    settings["timezoneInIndex"] = "UTC"
    settings["timezoneInRecord"] = "GMT"
    settings["timezoneInExecution"] = "America/Los_Angeles"
    settings["indexStartTime"] = "11:45:00"
    settings["indexEndTime"] = "20:00:00"
    settings["tradeStartTime"] = "11:45:00"
    settings["tradeEndTime"] = "20:00:00"
    settings["lunchStartTime"] = "11:55:00"
    settings["lunchEndTime"] = "11:55:00"
    settings["interval"] = 1
    settings["slippery"] = 0.2

    settings["dataPath"] = os.path.realpath("./data")
    settings["dateStart"] = "20190401"
    settings["dateEnd"] = "20190424"

    settings["graphOutput"] = "graph_output"
    settings["summaryOutput"] = "Summary.csv"

    return settings


if __name__ == "__main__":
    start = time.time()
    ## Define strategy settings here
    settings = dict()
    dataSettings = dict()
    logicSettings = OrderedDict()
    calSettings = dict()
    outputSettings = dict()

    settings = defaultSettings(settings)

    outputSettings["graphOutput"] = settings["graphOutput"]
    outputSettings["summaryOutput"] = settings["summaryOutput"]

    dataSettings["dataPath"] = settings["dataPath"]
    dataSettings["dateStart"] = settings["dateStart"]
    dataSettings["dateEnd"] = settings["dateEnd"]
    dataSettings["getDataDaysBefore"] = settings["dataPath"]

    # Add Logic here
    logicSettings[PnLLogic] = dict(minPnL=-500)

    logicSettings[BasicLogic] = dict(
        takeProfit=0.02, stopLoss=0.02, totalExposure=10, trailing=0
    )
    # logicSettings[RegLogic] = dict(
    #     period=475,
    #     delta_period=15,
    #     beta_threshold=0.0005,
    #     resamplePeriod=5,
    #     filter_type="momentum",
    # )
    # logicSettings[RSILogic] = dict(
    #     filterStart=10,
    #     filterEnd=52,
    #     filterWindow=372,
    #     # windowType = None,
    #     reverse=False,
    # )
    logicSettings[MALogic] = dict(
        fast_period=30,
        slow_period=120,
        wait_period=15,
        delta_thres=0.5,
        resample_period=5,
    )
    logicSettings[DayEndLogic] = dict(minsToForceExit=11.25)

    calSettings["slippery"] = settings["slippery"]

    strategy = Strategy(
        settings=settings,
        dataSettings=dataSettings,
        logicSettings=logicSettings,
        calSettings=calSettings,
        outputSettings=outputSettings,
    )
    # Use this for development
    strategy.evalStrat()
    # Use this for backtesting to use multiple cores
    # strategy.evalStratParallel()

    #   Output result to Summary.csv
    strategy.strategyOutput.outputToFile(strategy)
    #   Get realPnL for quick review
    realPnL = strategy.strategyOutput.getRealPnL(strategy)

    print("Real PnL for the strategy is: ", realPnL)

    # single-thread process for graph output
    # strategy.strategyOutput.outputGraphs(strategy, realtime=True)
    # multi-thread process for graph output
    strategy.outputGraphsParallel()

    print(
        "Benchmark Time from "
        + settings["dateStart"]
        + " to "
        + settings["dateEnd"]
        + " is "
        + str(time.time() - start)
        + " seconds."
    )
