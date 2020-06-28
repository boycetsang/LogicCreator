import numpy as np
import pandas as pd
import math
from statsmodels.regression.rolling import RollingOLS
from statsmodels.datasets import longley
from statsmodels.tools import add_constant


class Logic(object):
    """ Base Logic Object, sets up default behaviour for all Logic children """

    # Four boolean operations when appended with other logics
    # General recommendation is, if this logic is sufficient, then use 'or'. Otherwise, if condition is necessary, use 'and'
    BUY_ENTRANCE_LOGIC_OPERATION = "and"
    SELL_ENTRANCE_LOGIC_OPERATION = "and"
    BUY_EXIT_LOGIC_OPERATION = "and"
    SELL_EXIT_LOGIC_OPERATION = "and"

    def __init__(self):
        pass

    def addEntranceReport(self, strategy, position):
        return position

    def addExitReport(self, strategy, position):
        return position

    # Codes to determine condition for trades, if the function returns True it means such trade is allowed.
    def addBuyEntranceCondition(self, strategy, ind):
        return True

    def addBuyExitCondition(self, strategy, position, ind):
        return True

    def addSellEntranceCondition(self, strategy, ind):
        return True

    def addSellExitCondition(self, strategy, position, ind):
        return True

    # This code execute once for each day before iteration on time
    def computeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):
        return {}

    def computeForAction(self, strategy, ind, val):
        return {}

    # This code is used when producing graphical output
    def printOnSecondAxis(self, ax):
        return None


class DayEndLogic(Logic):
    """ Day End Logic, approaching end of day strategy should try to close positions
        This Logic class should be placed towards the end of logic chain.

        Parameters
        -----
        minToForceExit: int
            Time before tradeEndTime to force exit (in mins).

    """

    SELL_EXIT_LOGIC_OPERATION = "or"
    BUY_EXIT_LOGIC_OPERATION = "or"

    def __init__(self, minsToForceExit=30):
        self.minsToForceExit = minsToForceExit
        self.timeToForceExit = None

    def addSellExitCondition(self, strategy, position, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(
            minutes=self.minsToForceExit
        )
        if ind.time() >= self.timeToForceExit.time():
            return True
        else:
            return False

    def addBuyExitCondition(self, strategy, position, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(
            minutes=self.minsToForceExit
        )
        if ind.time() >= self.timeToForceExit.time():
            return True
        else:
            return False

    def addSellEntranceCondition(self, strategy, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(
            minutes=self.minsToForceExit
        )
        if ind.time() >= self.timeToForceExit.time():
            return False
        else:
            return True

    def addBuyEntranceCondition(self, strategy, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(
            minutes=self.minsToForceExit
        )
        if ind.time() >= self.timeToForceExit.time():
            return False
        else:
            return True


class BasicLogic(Logic):
    """ Standard Logic, controlling profit, loss and exposure
        
        Parameters
        -----
        takeProfit: float
            profit in points to terminate trade. If given a fractional number (< 1), 
            it is interpreted as relative to openPosition.
        stopLoss: float
            loss in points to terminate trade. If given a fractional number (< 1), 
            it is interpreted as relative to openPosition.
        totalExposure: int
            limit of number of concurrent trades.
        trailing: float
            if positive, trade termination by profit are not executed until index falls for this value from maximum.

        Returns at report
        -----
        totalExposure: int
            number of concurrent trades before this trade started.
    """

    def __init__(self, takeProfit=80, stopLoss=40, totalExposure=10, trailing=0):
        self.takeProfit = takeProfit
        self.stopLoss = stopLoss
        self.totalExposure = totalExposure
        self.trailing = trailing
        self.currentExposure = np.nan

    def addEntranceReport(self, strategy, position):
        position.totalExposure = self.currentExposure
        return position

    def addBuyEntranceCondition(self, strategy, ind):
        # Determine if exposure is too large
        if (
            strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt
            <= self.totalExposure
        ):
            self.currentExposure = (
                strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt
            )
            return True
        else:
            return False

    def addBuyExitCondition(self, strategy, position, ind):
        val = strategy.strategyData.timeSeriesTrade[ind]
        if self.takeProfit < 1:
            realTakeProfit = self.takeProfit * position.openPosition
        else:
            realTakeProfit = self.takeProfit
        if self.stopLoss < 1:
            realStopLoss = self.stopLoss * position.openPosition
        else:
            realStopLoss = self.stopLoss
        # Determine if price is outside range
        if (
            val > position.openPosition + realTakeProfit
            or val < position.openPosition - realStopLoss
        ):
            if self.trailing > 0:
                if val > position.openPosition + realTakeProfit:
                    tst = strategy.strategyData.timeSeriesTick
                    if tst[tst < position.openPosition + realTakeProfit].shape[0] > 0:
                        tst = tst[
                            tst[tst < position.openPosition + realTakeProfit].index[-1],
                            :,
                        ]
                        tst_max = tst.max()
                        if tst_max - val < self.trailing:
                            return False
            return True
        else:
            return False

    def addSellEntranceCondition(self, strategy, ind):
        # Determine if exposure is outside range
        if (
            strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt
            <= self.totalExposure
        ):
            self.currentExposure = (
                strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt
            )
            return True
        else:
            return False

    def addSellExitCondition(self, strategy, position, ind):
        if self.takeProfit < 1:
            realTakeProfit = self.takeProfit * position.openPosition
        else:
            realTakeProfit = self.takeProfit
        if self.stopLoss < 1:
            realStopLoss = self.stopLoss * position.openPosition
        else:
            realStopLoss = self.stopLoss

        val = strategy.strategyData.timeSeriesTrade[ind]
        if (
            val < position.openPosition - realTakeProfit
            or val > position.openPosition + realStopLoss
        ):
            if self.trailing > 0:
                if val < position.openPosition - realTakeProfit:
                    tst = strategy.strategyData.timeSeriesTick
                    tst = tst[
                        tst[tst > position.openPosition - realTakeProfit].index[-1] :
                    ]
                    tst_min = tst.min()
                    if -tst_min + val < self.trailing:
                        return False
            return True
        else:
            return False


class RSILogic(Logic):
    """ RSI Logic, controlling entrance using RSI indicator
        
        Parameters
        -----
        filterStart: float
            minimum RSI to trigger an entrance
        filterEnd: float
            maximum RSI to trigger an entrance
        filterWindow: float
            lag minutes to perform RSI calculation
        reverse: bool
            if True, reverse the behavior of the module

        Returns at report
        -----
        rsi: float
            RSI value at entrance point
    """
    def __init__(self, filterStart=10, filterEnd=60, filterWindow=30, reverse=False):
        self.filterStart = filterStart
        self.filterEnd = filterEnd
        self.filterWindow = filterWindow
        self.RSISeries = None
        # self.windowType = windowType
        self.reverse = reverse
        self.lastRSI = 0
        self.movement = None
        pass

    def addEntranceReport(self, strategy, position):
        position.rsi = self.lastRSI
        position.movement = self.movement
        return position

    def addExitReport(self, strategy, position):
        return position

    # Codes to determine condition for trades, if the function returns True it means such trade is allowed.
    def addBuyEntranceCondition(self, strategy, ind):
        RSICondition = False
        if any(self.RSISeries.index < ind):
            lastRSI = self.RSISeries[self.RSISeries.index < ind][-1]
            self.lastRSI = lastRSI
            RSICondition = lastRSI > self.filterEnd
            if self.reverse:
                RSICondition = not RSICondition
        return RSICondition

    def addSellEntranceCondition(self, strategy, ind):
        RSICondition = False
        if any(self.RSISeries.index < ind):
            lastRSI = self.RSISeries[self.RSISeries.index < ind][-1]
            self.lastRSI = lastRSI
            RSICondition = lastRSI < self.filterStart

            if self.reverse:
                RSICondition = not RSICondition
        return RSICondition

    # This code execute once for each day before iteration on time
    def computeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):
        window = self.filterWindow / float(
            strategy.strategySettings.settings["interval"]
        )
        # self.RSISeries = timeSeriesTick.rolling(window, win_type = self.windowType).std()
        close = timeSeriesTick
        delta = close.diff()
        # Get rid of the first row, which is NaN since it did not have a previous
        # row to calculate the differences
        delta = delta[1:]

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the EWMA
        # roll_up1 = pd.stats.moments.ewma(up, window)
        # roll_down1 = pd.stats.moments.ewma(down.abs(), window)
        roll_up1 = up.ewm(span=window).mean()
        roll_down1 = down.abs().ewm(span=window).mean()
        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        self.RSISeries = 100.0 - (100.0 / (1.0 + RS1))
        self.RSISeries = self.RSISeries[100:]
        return {"RSISeries": self.RSISeries}

    def computeForAction(self, strategy, ind, val):
        return {}

    # This code is used when producing graphical output
    def printOnSecondAxis(self, ax):
        ax.plot(self.RSISeries.index, self.RSISeries, color="b", alpha=0.7, label="rsi")
        ax.set_ylabel("rsi")
        ax.axhline(self.filterEnd, color="k", linestyle="--", lw=1, alpha=0.2)
        ax.axhline(self.filterStart, color="k", linestyle="--", lw=1, alpha=0.2)


class RegLogic(Logic):
    """ Trade filter using OLS regression

        Parameter
        -----
        period: int
            time window where OLS regression is performed (in seconds)
        delta_period: int
            time difference where the diff of beta is calculated (in seconds). Trade is only started if delta > 0 for buy, delta < 0 for sell.
        beta_threshold: float
            trades are not executed when beta is below this value
        resample_period: int
            resampling period of time series to perform OLS, the higher this number is the faster the regression is.
        filter_type: str ('momentum' or 'bounce')
            defines if trades are started opposite ('bounce') to trends or along ('momentum') with trends

        Return at report
        -----
        beta: float
            value of OLS beta when trade started
        delta: float
            value of diff of OLS when trade started
        
    """

    def __init__(
        self,
        period=600,
        delta_period=15,
        beta_threshold=0.03,
        resamplePeriod=5,
        filter_type="momentum",
    ):
        self.period = period
        self.delta_period = delta_period
        self.beta_threshold = beta_threshold
        self.resamplePeriod = resamplePeriod
        self.filter_type = filter_type
        self.betaSeries = None
        self.beta = None
        self.delta = None

    def addEntranceReport(self, strategy, position):
        position.beta = self.beta
        position.delta = self.delta
        return position

    def computeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):
        timeSeriesReg = timeSeriesTick.resample(
            str(int(self.resamplePeriod)) + "S"
        ).first()
        timeSeriesReg = timeSeriesReg.fillna(method="pad")
        timeTable = timeSeriesReg.to_frame()
        timeTable["second"] = timeSeriesReg.index.astype(np.int64)
        timeTable["second"] = (timeTable["second"] - timeTable["second"][0]) / math.pow(
            10, 9
        )

        # self.betaSeries = pd.stats.ols.MovingOLS(y=timeTable['price'], x=timeTable['second'], window_type='rolling', window = self.period, intercept=True).beta
        mod = RollingOLS(
            timeTable["price"],
            add_constant(timeTable["second"], prepend=False),
            window=self.period,
        )
        self.betaSeries = mod.fit().params
        return {"betaSeries": self.betaSeries}

    def addBuyEntranceCondition(self, strategy, ind):
        if any(self.betaSeries.index < ind):
            self.beta = self.betaSeries.second[self.betaSeries.index < ind][-1]
            self.delta = self.betaSeries.second.diff(periods=self.delta_period)[-1]
            if self.filter_type == "bounce":
                slopeCondition = (
                    self.betaSeries.second[self.betaSeries.index < ind][-1]
                    < -self.beta_threshold
                )
            else:
                slopeCondition = (
                    self.betaSeries.second[self.betaSeries.index < ind][-1]
                    > self.beta_threshold
                )
            deltaCondition = (
                self.betaSeries.second.diff(periods=self.delta_period)[-1] >= 0
            )
            if slopeCondition and deltaCondition:
                return True
            else:
                return False

    def addSellEntranceCondition(self, strategy, ind):
        return False
        if any(self.betaSeries.index < ind):
            if self.filter_type == "bounce":
                slopeCondition = (
                    self.betaSeries.second[self.betaSeries.index < ind][-1]
                    > self.beta_threshold
                )
            else:
                slopeCondition = (
                    self.betaSeries.second[self.betaSeries.index < ind][-1]
                    < -self.beta_threshold
                )
            deltaCondition = (
                self.betaSeries.second.diff(periods=self.delta_period)[-1] <= 0
            )
            if slopeCondition and deltaCondition:
                return True
            else:
                return False

    def printOnSecondAxis(self, ax):
        ax.plot(
            self.betaSeries.index,
            self.betaSeries.second,
            color="k",
            alpha=0.7,
            label="beta",
        )
        ax.set_ylabel("regressed slope (pt/sec)")
        ax.axhline(self.beta_threshold, color="k", linestyle="--", lw=1, alpha=0.2)
        ax.axhline(-self.beta_threshold, color="k", linestyle="--", lw=1, alpha=0.2)
        ax.fill_between(
            self.betaSeries.index,
            self.beta_threshold,
            self.betaSeries.second,
            where=self.betaSeries.second >= self.beta_threshold,
            alpha=0.2,
        )
        ax.fill_between(
            self.betaSeries.index,
            -self.beta_threshold,
            self.betaSeries.second,
            where=self.betaSeries.second <= -self.beta_threshold,
            alpha=0.2,
        )


class VelLogic(Logic):
    """ Trade filter using simple velocity

        Parameter
        -----
        period: int
            time window where velocity calculation is performed (in seconds)
        threshold: float 
            minimum velocity to trigger an entrance

        Return at report
        -----
        velocity: float
            value of velocity at entrance
        
    """
    def __init__(self, period=40, threshold=0.03):
        self.period = period
        self.threshold = threshold
        self.velocity = np.nan
        self.movement = None

    def addEntranceReport(self, strategy, position):
        position.velocity = self.velocity
        position.movement = self.movement
        return position

    def addBuyEntranceCondition(self, strategy, ind):
        tickInd = strategy.strategyData.timeSeriesTick.index
        timeToComputeVel = tickInd[tickInd < (ind - pd.Timedelta(seconds=self.period))]
        if len(timeToComputeVel) != 0:
            timeToComputeVel = timeToComputeVel[-1]
            self.velocity = (
                strategy.strategyData.timeSeriesTrade[ind]
                - strategy.strategyData.timeSeriesTick[timeToComputeVel]
            ) / (ind - timeToComputeVel).total_seconds()
            slopeCondition = self.velocity > self.threshold
        else:
            slopeCondition = False
        return slopeCondition

    def addSellEntranceCondition(self, strategy, ind):
        tickInd = strategy.strategyData.timeSeriesTick.index
        timeToComputeVel = tickInd[tickInd < (ind - pd.Timedelta(seconds=self.period))]
        if len(timeToComputeVel) != 0:
            timeToComputeVel = timeToComputeVel[-1]
            self.velocity = (
                strategy.strategyData.timeSeriesTrade[ind]
                - strategy.strategyData.timeSeriesTick[timeToComputeVel]
            ) / (ind - timeToComputeVel).total_seconds()
            slopeCondition = self.velocity < -self.threshold
        else:
            slopeCondition = False
        return slopeCondition


class PnLLogic(Logic):
    """ Trade filter using day-PnL

        Parameter
        -----
        minPnL: float
            if total PnL of the day is smaller than this value, no entrance can be triggered
        maxPnL: float
            if total PnL of the day is larger than this value, no entrance can be triggered
        Return at report
        -----
        minPnLInThisDayWhenOpen: float
            minimum PnL of the day at entrance point
        maxPnLInThisDayWhenOpen: float
            maximum PnL of the day at entrance point
        
    """
    SELL_EXIT_LOGIC_OPERATION = "or"
    BUY_EXIT_LOGIC_OPERATION = "or"

    def __init__(self, maxPnL=10000, minPnL=-10000):
        self.maxPnL = maxPnL
        self.minPnL = minPnL
        self.closedDates = []
        self.minPnLInThisDay = 0
        self.maxPnLInThisDay = 0
        pass

    def addEntranceReport(self, strategy, position):
        position.minPnLInThisDayWhenOpen = self.minPnLInThisDay
        position.maxPnLInThisDayWhenOpen = self.maxPnLInThisDay
        return position

    def addExitReport(self, strategy, position):
        ts = strategy.strategyData.timeSeriesTick
        if position.type == "buy":
            position.minPnLWithinTrade = (
                ts[(ts.index <= position.closeTime) & (ts.index >= position.openTime)]
                - position.openPosition
            ).min()
            position.maxPnLWithinTrade = (
                ts[(ts.index <= position.closeTime) & (ts.index >= position.openTime)]
                - position.openPosition
            ).max()
        elif position.type == "sell":
            position.minPnLWithinTrade = -(
                ts[(ts.index <= position.closeTime) & (ts.index >= position.openTime)]
                - position.openPosition
            ).max()
            position.maxPnLWithinTrade = -(
                ts[(ts.index <= position.closeTime) & (ts.index >= position.openTime)]
                - position.openPosition
            ).min()
        return position

    # Codes to determine condition for trades, if the function returns True it means such trade is allowed.
    def addBuyEntranceCondition(self, strategy, ind):
        if ind.date() in self.closedDates:
            return False
        else:
            return True

    def addBuyExitCondition(self, strategy, position, ind):
        if ind.date() in self.closedDates:
            return True
        else:
            return False

    def addSellEntranceCondition(self, strategy, ind):
        if ind.date() in self.closedDates:
            return False
        else:
            return True

    def addSellExitCondition(self, strategy, position, ind):
        if ind.date() in self.closedDates:
            return True
        else:
            return False

    # This code execute once for each day before iteration on time
    def computeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):
        self.minPnLInThisDay = 0
        self.maxPnLInThisDay = 0
        return {}

    def computeForAction(self, strategy, ind, val):
        cumPnL = 0
        if not ind.date() in self.closedDates:
            if strategy.strategyCalculator.positionList:
                pnlList = [
                    position.PnL
                    for position in strategy.strategyCalculator.positionList
                    if position.openTime.date() == ind.date()
                ]
            else:
                pnlList = []
            cumPnL = np.nansum(pnlList)
            if cumPnL >= self.maxPnL or cumPnL <= self.minPnL:
                self.closedDates.append(ind.date())
            else:
                if pnlList:
                    cumPnLArray = np.nancumsum(pnlList)
                    self.minPnLInThisDay = cumPnLArray.min()
                    self.maxPnLInThisDay = cumPnLArray.max()

    # This code is used when producing graphical output
    def printOnSecondAxis(self, ax):
        return None


class StdLogic(Logic):
    """ STD Logic, controlling entrance using standard deviation indicator. Standard deviation 
        calculations are de-trended by a OLS fit first
        
        Parameters
        -----
        filterStart: float
            minimum STD to trigger an entrance
        filterEnd: float
            maximum STD to trigger an entrance
        filterWindow: float
            lag minutes to perform STD calculation
        reverse: bool
            if True, reverse the behavior of the module

        Returns at report
        -----
        stdWhenOpen: float
            STD value at entrance point
    """
    def __init__(
        self,
        filterStart=10,
        filterEnd=60,
        filterWindow=30,
        windowType=None,
        reverse=False,
    ):
        self.filterStart = filterStart
        self.filterEnd = filterEnd
        self.filterWindow = filterWindow
        self.windowType = windowType
        self.reverse = reverse
        self.lastStd = 0
        pass

    def addEntranceReport(self, strategy, position):
        position.stdWhenOpen = self.lastStd
        return position

    def addExitReport(self, strategy, position):
        return position

    # Codes to determine condition for trades, if the function returns True it means such trade is allowed.
    def addBuyEntranceCondition(self, strategy, ind):
        stdCondition = True
        if any(self.stdSeries.index < ind):
            lastStd = self.stdSeries[self.stdSeries.index < ind][-1]
            self.lastStd = lastStd
            stdCondition = (lastStd < self.filterStart) | (lastStd > self.filterEnd)

            if self.reverse:
                stdCondition = not stdCondition
        return stdCondition

    def addSellEntranceCondition(self, strategy, ind):
        stdCondition = True
        if any(self.stdSeries.index < ind):
            lastStd = self.stdSeries[self.stdSeries.index < ind][-1]
            self.lastStd = lastStd
            stdCondition = (lastStd < self.filterStart) | (lastStd > self.filterEnd)

            if self.reverse:
                stdCondition = not stdCondition
        return stdCondition

    def detrendByBeta(self, ind_array):
        # Lazy import because this is not used often
        import scipy.stats as stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.array(range(len(ind_array))), ind_array
        )
        ind_array = ind_array - range(len(ind_array)) * slope
        return ind_array.std()

    # This code execute once for each day before iteration on time
    def computeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):

        window = str(int(self.filterWindow)) + "T"
        timeSeriesReg = timeSeriesTick.resample(
            str(int(self.filterWindow / 10)) + "S"
        ).first()
        timeSeriesReg = timeSeriesReg.fillna(method="pad")
        # self.stdSeries = timeSeriesTick.rolling(window, win_type = self.windowType).std()
        self.stdSeries = timeSeriesReg.rolling(window, win_type=None).apply(
            self.detrendByBeta
        )
        return {"stdSeries": self.stdSeries}

    def computeForAction(self, strategy, ind, val):
        return {}

    # This code is used when producing graphical output
    def printOnSecondAxis(self, ax):
        ax.plot(self.stdSeries.index, self.stdSeries, color="r", alpha=0.7)
        ax.set_ylabel("rolling std")
        ax.axhline(self.filterEnd, color="k", linestyle="--", lw=2, alpha=0.2)
        ax.axhline(self.filterStart, color="k", linestyle="--", lw=2, alpha=0.2)


class MALogic(Logic):
    """ Logic to capture moving average corssing behavior in time series.
        This Logic does not execute short positions
        
        Parameters
        -----
        fast_period: int
            time window which OLS regression is performed (in seconds)
        slow_period: int
            time window which moving average of beta time series is calculated (in seconds)
        delta_thres: float
            minimum amount of excess flip to trigger entance execution
        wait_period: int
            time window in which the MA flip has to occur to trigger execution
        resample_period: int
            time window to interpolate for missing values. This value cannot be samller than any of above.
    """

    SELL_EXIT_LOGIC_OPERATION = "or"
    BUY_EXIT_LOGIC_OPERATION = "or"

    def __init__(
        self,
        fast_period=480,
        slow_period=180,
        delta_thres=0,
        wait_period=20,
        resample_period=5,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.delta_thres = delta_thres
        self.wait_period = wait_period
        self.resample_period = resample_period

    def addEntranceReport(self, strategy, position):
        position.ewma = self.ewma
        position.ewmaLong = self.ewmaLong

    def computeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):
        timeSeriesReg = timeSeriesTick.resample(
            str(int(self.resample_period)) + "S"
        ).first()
        timeSeriesReg = timeSeriesReg.fillna(method="pad")
        timeTable = timeSeriesReg.to_frame()
        # timeTable = timeSeriesReg
        timeTable["second"] = timeSeriesReg.index.astype(np.int64)
        timeTable["second"] = (timeTable["second"] - timeTable["second"][0]) / math.pow(
            10, 9
        )

        self.ewmaSeries = (
            timeSeriesReg.ewm(span=self.fast_period / self.resample_period)
            .mean()
            .rename("ewma")
        )
        self.ewmaLongSeries = (
            timeSeriesReg.ewm(span=self.slow_period / self.resample_period)
            .mean()
            .rename("ewmaLong")
        )
        self.ewma = self.ewmaSeries.iloc[-1]
        self.ewmaLong = self.ewmaLongSeries.iloc[-1]
        return {"ewmaSeries": self.ewmaSeries, "ewmaLongSeries": self.ewmaLongSeries}

    def addBuyExitCondition(self, strategy, position, ind):
        wait_ind = (
            self.ewmaSeries.index < ind + pd.Timedelta(self.wait_period, unit="s")
        ) & (self.ewmaSeries.index >= ind)
        if (self.ewmaSeries - self.ewmaLongSeries)[wait_ind].max() < 0 and (
            (self.ewmaSeries - self.ewmaLongSeries)[wait_ind].argmin()
            == (len((self.ewmaSeries - self.ewmaLongSeries)[wait_ind]) - 1)
        ):
            return True
        else:
            return False

    def addSellExitCondition(self, strategy, position, ind):
        return False

    def addBuyEntranceCondition(self, strategy, ind):
        wait_ind = (
            self.ewmaSeries.index < ind + pd.Timedelta(self.wait_period, unit="s")
        ) & (self.ewmaSeries.index >= ind)
        if (self.ewmaSeries - self.ewmaLongSeries)[wait_ind].min() > self.delta_thres:
            return True
        else:
            return False

    def addSellEntranceCondition(self, strategy, ind):
        return False

    def printOnSecondAxis(self, ax):
        ax.plot(
            self.ewmaSeries.index, self.ewmaSeries, color="b", alpha=0.7, label="ewma"
        )
        ax.plot(
            self.ewmaLongSeries.index,
            self.ewmaLongSeries,
            color="r",
            alpha=0.7,
            label="ewmaLong",
        )
