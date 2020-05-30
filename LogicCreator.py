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


def defaultSettings(settings):
    settings["timezoneInIndex"] = "UTC"
    settings["timezoneInRecord"] = "GMT"
    settings["indexStartTime"] = "11:45:00"
    settings["indexEndTime"] = "20:00:00"
    settings["tradeStartTime"] = "11:45:00"
    settings["tradeEndTime"] = "20:00:00"
    settings["lunchStartTime"] = "11:55:00"
    settings["lunchEndTime"] = "11:55:00"
    settings["interval"] = 1
    settings["slippery"] = 0.5

    settings["dataPath"] = r"/mnt/c/Users/Boyce/PythonProject/LogicCreator/data"
    settings["dateStart"] = "20190422"
    settings["dateEnd"] = "20190424"

    settings["graphOutput"] = "graph_output"
    settings["summaryOutput"] = "Summary.csv"

    return settings


class Strategy:
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
        timeSeries["datetime"] = pd.to_datetime(timeSeries.tradeTime)
        timeSeries = timeSeries.set_index("datetime")
        timeSeries.index = timeSeries.index.tz_localize(
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
        # print(timeSeries.index.min(), timeSeries.index.max())

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
        # treating the insufficient data case
        if len(timeSeries) != 0:
            timeSeries[self.indexEndTime] = timeSeries[
                (timeSeries.index <= self.indexEndTime)
            ].iloc[-1]
        # print(timeSeries)
        return timeSeries

    def evalStrat(self, silence=False):
        # evaluate strategy for every day defined
        date_cnt = 0
        date_total = len(
            set(self.strategyData.tradeDateRange.date).intersection(
                set(self.strategyData.dateFileDict.keys())
            )
        )
        if not silence:
            print("Backtesting in progress ...")
        for date in self.strategyData.tradeDateRange:
            date = date.date()
            if not date in self.strategyData.dateFileDict.keys():
                continue
            self.strategyData.currentDate = date
            self.strategyData.timeSeries = pd.read_csv(
                self.strategyData.dateFileDict[date], na_values=["NA"]
            )

            self.strategyData.timeSeries = self.dataClean(
                date, self.strategyData.timeSeries
            )

            self.strategyData.timeSeriesTick = self.strategyData.timeSeries
            self.strategyData.timeSeriesTrade = self.strategyData.timeSeries.resample(
                str(int(float(self.strategySettings.settings["interval"]) * 60)) + "S"
            ).first()
            if self.strategyData.timeSeriesTrade.empty:
                date_cnt += 1
                continue
            self.strategyLogic.performComputeForDay(
                self,
                self.strategyData.timeSeriesTick,
                self.strategyData.timeSeriesTrade,
            )

            for ind, val in self.strategyData.timeSeriesTrade.iteritems():
                self.strategyLogic.performComputeForAction(self, ind, val)
                if np.all(np.isnan(val)):
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
            date_cnt += 1
            if not silence:
                self.strategyOutput.updateProgress(date_cnt / date_total)

    def evalStratDay(self, date):
        date = date.date()
        if not date in self.strategyData.dateFileDict.keys():
            return
        self.strategyData.currentDate = date
        self.strategyData.timeSeries = pd.read_csv(
            self.strategyData.dateFileDict[date], na_values=["NA"]
        )
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

        result = copy.deepcopy(self.strategyCalculator.positionList)
        self.strategyCalculator.positionList = []
        return result

    def evalStratParallel(self, silence=False, num_core=0):
        # evaluate strategy for every day defined
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
        # produce graph for every day defined
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

    #    strategy.strategySettings = StrategySettings(settings)
    dataSettings["dataPath"] = settings["dataPath"]
    dataSettings["dateStart"] = settings["dateStart"]
    dataSettings["dateEnd"] = settings["dateEnd"]
    dataSettings["getDataDaysBefore"] = settings["dataPath"]

    # Add Logic here
    #    logicSettings[BasicLogic] = {'takeProfit' : settings['Basic_takeProfit'], "stopLoss" : settings['Basic_stopLoss'], 'totalExposure': settings['Basic_maxExposure']}
    #    logicSettings[VelLogic] = {'period': settings['Vel_period'], 'threshold': settings['Vel_threshold']}
    #    logicSettings[RegLogic] = {'period': settings['Reg_period'], 'delta_period': settings['Reg_delta_period'], 'beta_threshold': settings['Reg_beta_threshold'], 'filter_type': settings['Reg_filter_type']}

    #    logicSettings[StdLogic] = {'filterStart': settings['std_filterStart'], 'filterEnd': settings['std_filterEnd'], 'filterWindow': settings['std_filterWindow']}
    #    logicSettings[PnLLogic] = {'minPnL': settings['PnL_minPnL'], 'maxPnL': settings['PnL_maxPnL']}
    # logicSettings[DayEndLogic] = {'minsToForceExit': settings['DayEnd_minsToForceExit']}
    #    logicSettings[RyanLogic] = {'x': 10, 'y': 200}
    #    print(logicSettings)

    # ogicSettings[BasicLogic] = {}
    # logicSettings[VelLogic] = {}
    #    logicSettings[RegLogic] = {}

    #    logicSettings[StdLogic] = {}
    #    logicSettings[PnLLogic] = {}

    logicSettings[BasicLogic] = dict(
        takeProfit=7.65, stopLoss=7.65, totalExposure=10, trailing=0
    )
    logicSettings[RegLogic] = dict(
        period=475,
        delta_period=15,
        beta_threshold=0.0005,
        resamplePeriod=5,
        filter_type="momentum",
    )
    logicSettings[RSILogic] = dict(
        filterStart=10,
        filterEnd=52,
        filterWindow=372,
        # windowType = None,
        reverse=False,
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

    # strategy.strategyOutput.outputGraphs(strategy)
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
