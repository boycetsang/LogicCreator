import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class StrategyOutput:
    def __init__(self, outputSettings):
        self.report = pd.DataFrame()
        self.outputSettings = outputSettings
        self.outputTable = pd.DataFrame()
        self.lastAxes = None

    def __str__(self):
        return "StrategyOutput: " + str(self.outputSettings)

    def pushToFirst(self, lst, elements):
        if isinstance(elements, str):
            elements = [elements]
        for element in reversed(elements):
            if element in lst:
                lst.remove(element)
                lst.insert(0, element)
        return lst

    def outputToFile(self, strategy, silence=False, useNumber=0,):
        outputTable = strategy.strategyCalculator.printPositioninTable()
        if not outputTable.empty:
            # do some organization ...
            outputTable.closeTime = pd.to_datetime(
                outputTable.closeTime, format="%Y%m%d %H:%M:%S.%f"
            )
            outputTable["closeDate"] = outputTable.closeTime.dt.strftime("%Y%m%d")
            outputTable.closeTime = outputTable.closeTime.dt.time
            outputTable.openTime = pd.to_datetime(
                outputTable.openTime, format="%Y%m%d %H:%M:%S.%f"
            )
            outputTable["openDate"] = outputTable.openTime.dt.strftime("%Y%m%d")
            outputTable.openTime = outputTable.openTime.dt.time

            # rearrange columns
            cols = outputTable.columns.tolist()
            openCols = ["openDate", "openTime", "openPosition"]
            closeCols = ["closeDate", "closeTime", "closePosition"]
            pushCols = openCols + closeCols
            cols = self.pushToFirst(cols, pushCols)
            cols = self.pushToFirst(cols, "type")
            outputTable = outputTable.reindex(columns=cols)
            self.outputTable = outputTable

            if not silence:
                if useNumber:
                    outputTable.to_csv(
                        "SOLN_OUTPUT_" + str(useNumber) + ".csv", index=None
                    )
                else:
                    outputTable.to_csv(self.outputSettings["summaryOutput"], index=None)
                # print()
                # print('Finished writing summary file to ' + self.outputSettings['summaryOutput'])
        else:
            self.outputTable = pd.DataFrame()

    def updateProgress(self, progress):
        """ Simple progress bar"""
        sys.stdout.write(
            "\r[{0}] {1:.0f}%".format(
                "#" * int(progress * 10) + "." * int(10 - progress * 10), progress * 100
            )
        )

    def outputGraphs(self, strategy, dates=[], data=None, realtime=0):
        import matplotlib.dates as mdates

        if dates == [] and self.outputSettings["graphOutput"]:
            import pathlib

            pathlib.Path(self.outputSettings["graphOutput"]).mkdir(
                parents=True, exist_ok=True
            )
            for date in strategy.strategyData.tradeDateRange:
                date = date.date()
                if data is None and not date in strategy.strategyData.dateFileDict.keys():
                    continue
                if data is not None:
                    timeSeries = data
                else:
                    timeSeries = pd.read_csv(
                        strategy.strategyData.dateFileDict[date],
                        index_col=None,
                        na_values=["NA"],
                    )
                timeSeries = strategy.dataClean(date, timeSeries)

                timeSeriesTick = timeSeries

                timeSeriesTrade = timeSeries.resample(
                    str(int(float(strategy.strategySettings.settings["interval"]) * 60))
                    + "S"
                ).first()

                if timeSeriesTrade.empty:
                    continue

                # timeSeriesTick.rename("price_tick").plot()
                # timeSeriesTrade.rename("price_interval").plot()


                strategy.strategyLogic.performComputeForDay(
                    strategy, timeSeriesTick, timeSeriesTrade
                )
                ax1 = plt.gca()
                ax1.plot(timeSeriesTick.index, timeSeriesTick, label="price_tick")
                ax1.plot(timeSeriesTrade.index, timeSeriesTrade, label="price_interval")
                ax1.set_ylabel("price")
                ax2 = ax1.twinx()
                for logicSetting in strategy.strategyLogic.logicCollection:
                    logicSetting.printOnSecondAxis(ax2)
                plt.sca(ax1)
                ax1.legend(loc=3)
                ax2.legend(loc=4)
                positionListForThisDate = [
                    position
                    for position in strategy.strategyCalculator.positionList
                    if position.openTime.date() == date
                ]
                for position in positionListForThisDate:
                    if position.type == "buy":
                        if hasattr(position, "movement") and position.movement:
                            if position.closePosition > position.openPosition:
                                plt.plot(
                                    [position.openTime, position.openTime],
                                    [
                                        position.openPosition,
                                        position.openPosition + position.movement,
                                    ],
                                    "-.g",
                                )
                            else:
                                plt.plot(
                                    [position.openTime, position.openTime],
                                    [
                                        position.openPosition,
                                        position.openPosition + position.movement,
                                    ],
                                    "-.r",
                                )
                        if position.closePosition > position.openPosition:
                            plt.plot(position.openTime, position.openPosition, "og")
                            plt.plot(
                                [position.openTime, position.closeTime],
                                [position.openPosition, position.closePosition],
                                "--g",
                                linewidth=1,
                            )
                        else:
                            if np.isnan(position.closePosition):
                                plt.plot(position.openTime, position.openPosition, "ok")
                            else:
                                plt.plot(position.openTime, position.openPosition, "or")
                            try:
                                plt.plot(
                                    [position.openTime, position.closeTime],
                                    [position.openPosition, position.closePosition],
                                    "--r",
                                    linewidth=1,
                                )
                            except:
                                pass
                    if position.type == "sell":
                        if hasattr(position, "movement") and position.movement:
                            if position.closePosition > position.openPosition:
                                plt.plot(
                                    [position.openTime, position.openTime],
                                    [
                                        position.openPosition,
                                        position.openPosition - position.movement,
                                    ],
                                    "-.r",
                                )
                            else:
                                plt.plot(
                                    [position.openTime, position.openTime],
                                    [
                                        position.openPosition,
                                        position.openPosition - position.movement,
                                    ],
                                    "-.g",
                                )
                        if position.closePosition < position.openPosition:
                            plt.plot(position.openTime, position.openPosition, "sg")
                            plt.plot(
                                [position.openTime, position.closeTime],
                                [position.openPosition, position.closePosition],
                                "--g",
                                linewidth=1,
                            )
                        else:
                            if np.isnan(position.closePosition):
                                plt.plot(position.openTime, position.openPosition, "ok")
                            else:
                                plt.plot(position.openTime, position.openPosition, "sr")
                            plt.plot(
                                [position.openTime, position.closeTime],
                                [position.openPosition, position.closePosition],
                                "--r",
                                linewidth=1,
                            )

                graphPath = os.path.join(
                    os.getcwd(), self.outputSettings["graphOutput"], str(date) + ".png"
                )
                xfmt = mdates.DateFormatter("%H:%M")
                plt.gca().xaxis.set_major_formatter(xfmt)
                plt.savefig(graphPath, dpi=300)
                if realtime:
                    plt.show(block=False)
                    plt.pause(realtime)
                    self.lastAxes = plt.axis()
                    # plt.close()
                plt.clf()
            # print("Finished writing graphs to " + self.outputSettings["graphOutput"])

    def outputGraphsDay(self, strategy, date):
        import matplotlib.dates as mdates

        if self.outputSettings["graphOutput"]:
            import pathlib

            pathlib.Path(self.outputSettings["graphOutput"]).mkdir(
                parents=True, exist_ok=True
            )
            date = date.date()

            if not date in strategy.strategyData.dateFileDict.keys():
                return
            timeSeries = pd.read_csv(
                strategy.strategyData.dateFileDict[date],
                index_col=None,
                na_values=["NA"],
            )
            timeSeries = strategy.dataClean(date, timeSeries)
            timeSeriesTick = timeSeries
            timeSeriesTrade = timeSeries.resample(
                str(int(float(strategy.strategySettings.settings["interval"]) * 60))
                + "S"
            ).first()

            if timeSeriesTrade.empty:
                return
            timeSeriesTick.rename("price_tick").plot()
            timeSeriesTrade.rename("price_interval").plot()
            print(timeSeriesTick.index)
            strategy.strategyLogic.performComputeForDay(
                strategy, timeSeriesTick, timeSeriesTrade
            )
            ax1 = plt.gca()
            ax1.set_ylabel("index")
            ax2 = ax1.twinx()
            for logicSetting in strategy.strategyLogic.logicCollection:
                logicSetting.printOnSecondAxis(ax2)
            plt.sca(ax1)
            ax1.legend(loc=3)
            ax2.legend(loc=4)

            positionListForThisDate = [
                position
                for position in strategy.strategyCalculator.positionList
                if position.openTime.date() == date
            ]
            for position in positionListForThisDate:
                if position.type == "buy":
                    if position.movement:
                        if position.closePosition > position.openPosition:
                            plt.plot(
                                [position.openTime, position.openTime],
                                [
                                    position.openPosition,
                                    position.openPosition + position.movement,
                                ],
                                "-.g",
                            )
                        else:
                            plt.plot(
                                [position.openTime, position.openTime],
                                [
                                    position.openPosition,
                                    position.openPosition + position.movement,
                                ],
                                "-.r",
                            )
                    if position.closePosition > position.openPosition:
                        plt.plot(position.openTime, position.openPosition, "og")
                        plt.plot(
                            [position.openTime, position.closeTime],
                            [position.openPosition, position.closePosition],
                            "--g",
                            linewidth=1,
                        )
                    else:
                        plt.plot(position.openTime, position.openPosition, "or")
                        plt.plot(
                            [position.openTime, position.closeTime],
                            [position.openPosition, position.closePosition],
                            "--r",
                            linewidth=1,
                        )
                if position.type == "sell":
                    if position.movement:
                        if position.closePosition > position.openPosition:
                            plt.plot(
                                [position.openTime, position.openTime],
                                [
                                    position.openPosition,
                                    position.openPosition - position.movement,
                                ],
                                "-.r",
                            )
                        else:
                            plt.plot(
                                [position.openTime, position.openTime],
                                [
                                    position.openPosition,
                                    position.openPosition - position.movement,
                                ],
                                "-.g",
                            )
                    if position.closePosition < position.openPosition:
                        plt.plot(position.openTime, position.openPosition, "sg")
                        plt.plot(
                            [position.openTime, position.closeTime],
                            [position.openPosition, position.closePosition],
                            "--g",
                            linewidth=1,
                        )
                    else:
                        plt.plot(position.openTime, position.openPosition, "sr")
                        plt.plot(
                            [position.openTime, position.closeTime],
                            [position.openPosition, position.closePosition],
                            "--r",
                            linewidth=1,
                        )

            graphPath = os.path.join(
                os.getcwd(), self.outputSettings["graphOutput"], str(date) + ".png"
            )
            xfmt = mdates.DateFormatter("%H:%M")
            plt.gca().xaxis.set_major_formatter(xfmt)
            plt.savefig(graphPath, dpi=300)
            plt.clf()

    def getPnL(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            return self.outputTable.PnL.sum()

    def getRealPnL(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            return self.outputTable.realPnL.sum()

    def getRealPnLPerTradedDay(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            realPnLPerTradedDay = (
                self.outputTable.groupby("openDate")
                .apply(lambda x: x.realPnL.sum())
                .mean()
            )
            return realPnLPerTradedDay

    def getRealPnLPerDay(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            realPnLPerDay = (
                self.outputTable.realPnL.sum()
                / (strategy.strategyData.dateEnd - strategy.strategyData.dateStart).days
            )
            return realPnLPerDay

    def getRealPnLStdDay(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            realPnLPerStddDay = (
                self.outputTable.groupby("openDate")
                .apply(lambda x: x.realPnL.sum())
                .std()
            )
            return realPnLPerStddDay

    def getRealPnLStdMonth(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            self.outputTable["openMonth"] = pd.to_datetime(
                self.outputTable.openDate, format="%Y%m%d"
            ).apply(lambda x: x.month)

            realPnLPerStddDay = (
                self.outputTable.groupby("openMonth")
                .apply(lambda x: x.realPnL.sum())
                .std()
            )
            return realPnLPerStddDay

    def getRealPnLPerTrade(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            return self.outputTable.realPnL.mean()

    def getRealPnLMaxDrawBackInDay(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            realPnLPerMaxDrawBackInDay = self.outputTable.groupby("openDate").apply(
                lambda x: (
                    np.nancumsum(x.realPnL.iloc[::-1])
                    - pd.Series(np.nancumsum(x.realPnL.iloc[::-1])).cummax()
                ).min()
            )
            return realPnLPerMaxDrawBackInDay.min()

    def getRealPnLMaxDrawBack(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            realPnLPerMaxDrawBack = (
                np.nancumsum(self.outputTable.realPnL.iloc[::-1])
                - pd.Series(np.nancumsum(self.outputTable.realPnL.iloc[::-1])).cummax()
            ).min()
            return realPnLPerMaxDrawBack

    def getRealWinDayRatio(self, strategy):
        if self.outputTable.empty:
            self.outputToFile(strategy, silence=True)
        if self.outputTable.empty:
            return 0
        else:
            winDayRatio = (
                self.outputTable.groupby("openDate").apply(
                    lambda x: x.realPnL.sum() >= 0
                )
            ).sum() / len(self.outputTable["openDate"].value_counts())
            return winDayRatio
