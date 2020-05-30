import numpy as np
import pandas as pd


class StrategyCalculator:
    def __init__(self, calSettings):
        self.calSettings = calSettings
        self.buyCnt = 0
        self.sellCnt = 0
        self.positionList = []
        self.openPositionList = []

    def __str__(self):
        return os.linesep.join(
            ["StrategyCalculator: " + str(self.calSettings)]
            + ["Current Positions: "]
            + [str(position) for position in self.openPositionList]
        )

    def printPositioninTable(self):
        df = pd.DataFrame()

        if self.positionList:
            position_dict_list = [vars(position) for position in self.positionList]
        else:
            self.positionList = []
            return pd.DataFrame()
        df = pd.DataFrame(position_dict_list)
        if "strategy" in df.columns.values:
            del df["strategy"]

        return df

    class Position(object):
        def __init__(self, strategy, openTime, openPosition, type):
            self.strategy = strategy
            self.openTime = openTime
            self.openPosition = openPosition
            self.closeTime = np.nan
            self.closePosition = np.nan
            self.tradeAmount = 1
            self.PnL = np.nan
            self.realPnL = np.nan
            self.type = type
            self.buyCntWhenOpen = self.strategy.strategyCalculator.buyCnt
            self.sellCntWhenOpen = self.strategy.strategyCalculator.sellCnt
            if type == "buy":
                self.strategy.strategyCalculator.buyCnt += 1
            if type == "sell":
                self.strategy.strategyCalculator.sellCnt += 1
            self.strategy.strategyCalculator.positionList += [self]
            self.strategy.strategyCalculator.openPositionList += [self]

        def __str__(self):
            return os.linesep.join(
                [
                    "Position: " + str(self.type),
                    "Open at time " + str(self.openTime),
                    "Open at price " + str(self.openPosition),
                    "Close at time " + str(self.closeTime),
                    "Close at price " + str(self.closePosition),
                ]
            )

        def close(self, closeTime, closePosition):
            self.closeTime = closeTime
            self.closePosition = closePosition
            if self.type == "buy":
                self.strategy.strategyCalculator.buyCnt -= 1
                self.PnL = (closePosition - self.openPosition) * self.tradeAmount
                self.realPnL = (
                    self.PnL - self.strategy.strategyCalculator.calSettings["slippery"]
                )
            if self.type == "sell":
                self.strategy.strategyCalculator.sellCnt -= 1
                self.PnL = (-closePosition + self.openPosition) * self.tradeAmount
                self.realPnL = (
                    self.PnL - self.strategy.strategyCalculator.calSettings["slippery"]
                )

            self.strategy.strategyCalculator.openPositionList.remove(self)
