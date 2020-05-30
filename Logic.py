


class Logic(object):
    ''' Base Logic Object, sets up default behaviour for all Logic children '''
    # Four boolean operations when appended with other logics
    # General recommendation is, if this logic is sufficient, then use 'or'. Otherwise, if condition is necessary, use 'and'
    BUY_ENTRANCE_LOGIC_OPERATION = 'and'
    SELL_ENTRANCE_LOGIC_OPERATION = 'and'
    BUY_EXIT_LOGIC_OPERATION = 'and'
    SELL_EXIT_LOGIC_OPERATION = 'and'    
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
    ''' Day End Logic, approaching end of day strategy should try to close positions
        This Logic class should be placed towards the end of logic chain.

        Parameters
        -----
        minToForceExit: int
            Time before tradeEndTime to force exit (in mins).

    '''
    SELL_EXIT_LOGIC_OPERATION = 'or'
    BUY_EXIT_LOGIC_OPERATION = 'or'
    def __init__(self, minsToForceExit = 30):
        self.minsToForceExit = minsToForceExit
        self.timeToForceExit = None
    
    def addSellExitCondition(self, strategy, position, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(minutes=self.minsToForceExit)
        if ind.time() >= self.timeToForceExit.time():
            return True
        else:
            return False
    def addBuyExitCondition(self, strategy, position, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(minutes=self.minsToForceExit)
        if ind.time() >= self.timeToForceExit.time():
            return True
        else:
            return False
    def addSellEntranceCondition(self, strategy, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(minutes=self.minsToForceExit)
        if ind.time() >= self.timeToForceExit.time():
            return False
        else:
            return True
    def addBuyEntranceCondition(self, strategy, ind):
        self.timeToForceExit = strategy.tradeEndTime - pd.Timedelta(minutes=self.minsToForceExit)
        if ind.time() >= self.timeToForceExit.time():
            return False
        else:
            return True


class BasicLogic(Logic):
    ''' Standard Logic, controlling profit, loss and exposure
        
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
    '''
    def __init__(self, takeProfit = 80, stopLoss = 40, totalExposure = 10, trailing = 0):
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
        if (strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt <= self.totalExposure):
            self.currentExposure = strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt
            return True
        else:
            return False
    def addBuyExitCondition(self, strategy, position, ind):
        val = strategy.strategyData.timeSeriesTrade[ind]
        if self.takeProfit < 1:
            realTakeProfit = self.takeProfit * position.openPosition
        else:
            realTakeProfit = self.takeProfit 
        if self.stopLoss  < 1:
            realStopLoss = self.stopLoss * position.openPosition
        else:
            realStopLoss = self.stopLoss
        # Determine if price is outside range
        if val > position.openPosition + realTakeProfit or val < position.openPosition - realStopLoss:
            if self.trailing > 0:
                if val > position.openPosition + realTakeProfit :
                    tst = strategy.strategyData.timeSeriesTick
                    if tst[tst < position.openPosition + realTakeProfit].shape[0] > 0:
                        tst = tst[tst[tst < position.openPosition + realTakeProfit].index[-1]:]
                        tst_max = tst.max()
                        if tst_max - val < self.trailing:
                            return False
            return True
        else:
            return False
        
    def addSellEntranceCondition(self, strategy, ind):
        # Determine if exposure is outside range
        if (strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt <= self.totalExposure):
            self.currentExposure = strategy.strategyCalculator.buyCnt + strategy.strategyCalculator.sellCnt
            return True
        else:
            return False
        
    def addSellExitCondition(self, strategy, position, ind):
        if self.takeProfit < 1:
            realTakeProfit = self.takeProfit * position.openPosition
        else:
            realTakeProfit = self.takeProfit 
        if self.stopLoss  < 1:
            realStopLoss = self.stopLoss * position.openPosition
        else:
            realStopLoss = self.stopLoss
    
        val = strategy.strategyData.timeSeriesTrade[ind]
        if val < position.openPosition - realTakeProfit or val > position.openPosition + realStopLoss:
            if self.trailing > 0:
                if val < position.openPosition - realTakeProfit :
                    tst = strategy.strategyData.timeSeriesTick
                    tst = tst[tst[tst > position.openPosition - realTakeProfit].index[-1]:]
                    tst_min = tst.min()
                    if -tst_min + val < self.trailing:
                        return False
            return True
        else:
            return False


class RSILogic(Logic):
    def __init__(self, filterStart = 10, filterEnd = 60, filterWindow = 30, reverse = False):
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
            RSICondition =  (lastRSI > self.filterEnd)
            if self.reverse:
                RSICondition = not RSICondition
        return RSICondition

    def addSellEntranceCondition(self, strategy, ind):
        RSICondition = False
        if any(self.RSISeries.index < ind):
            lastRSI = self.RSISeries[self.RSISeries.index < ind][-1]
            self.lastRSI = lastRSI
            RSICondition = (lastRSI < self.filterStart) 

            if self.reverse:
                RSICondition = not RSICondition
        return RSICondition

    # This code execute once for each day before iteration on time
    def computeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):
        window = self.filterWindow/float(strategy.strategySettings.settings['interval'])
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
        return {'RSISeries': self.RSISeries}

    def computeForAction(self, strategy, ind, val):
        return {}
    # This code is used when producing graphical output
    def printOnSecondAxis(self, ax):
        ax.plot(self.RSISeries.index, self.RSISeries, color = 'b', alpha = 0.7, label='rsi')
        ax.set_ylabel('rsi')
        ax.axhline(self.filterEnd, color='k', linestyle='--', lw=1, alpha = 0.2)
        ax.axhline(self.filterStart, color='k', linestyle='--', lw=1, alpha = 0.2)


class RegLogic(Logic):
    ''' Trade filter using OLS regression

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
        
    '''
    def __init__(self, period = 600, delta_period = 15, beta_threshold = 0.03, resamplePeriod = 5, filter_type = 'momentum'):
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
        timeSeriesReg = timeSeriesTick.resample(str(int(self.resamplePeriod))+'S').first()
        timeSeriesReg = timeSeriesReg.fillna(method='pad')
        timeTable = timeSeriesReg.to_frame()
        timeTable['second'] = timeSeriesReg.index.astype(np.int64)
        timeTable['second'] = (timeTable['second'] - timeTable['second'][0])/math.pow(10,9)

        # self.betaSeries = pd.stats.ols.MovingOLS(y=timeTable['price'], x=timeTable['second'], window_type='rolling', window = self.period, intercept=True).beta
        mod = RollingOLS(timeTable['price'], add_constant(timeTable['second'], prepend=False), window=self.period)
        self.betaSeries = mod.fit().params
        return {'betaSeries': self.betaSeries}
    
    def addBuyEntranceCondition(self, strategy, ind):
        if any(self.betaSeries.index < ind):
            self.beta = self.betaSeries.second[self.betaSeries.index < ind][-1]
            self.delta = self.betaSeries.second.diff(periods = self.delta_period)[-1]
            if self.filter_type == 'bounce':
                slopeCondition = self.betaSeries.second[self.betaSeries.index < ind][-1] < -self.beta_threshold
            else:
                slopeCondition = self.betaSeries.second[self.betaSeries.index < ind][-1] > self.beta_threshold
            deltaCondition = self.betaSeries.second.diff(periods = self.delta_period)[-1] >= 0
            if slopeCondition and deltaCondition:
                return True
            else:
                return False
        
    def addSellEntranceCondition(self, strategy, ind):
        return False
        if any(self.betaSeries.index < ind):
            if self.filter_type == 'bounce':
                slopeCondition = self.betaSeries.second[self.betaSeries.index < ind][-1] > self.beta_threshold
            else:
                slopeCondition = self.betaSeries.second[self.betaSeries.index < ind][-1] < -self.beta_threshold
            deltaCondition = self.betaSeries.second.diff(periods = self.delta_period)[-1] <= 0
            if slopeCondition and deltaCondition:
                return True
            else:
                return False
    def printOnSecondAxis(self, ax):
        ax.plot(self.betaSeries.index, self.betaSeries.second, color = 'k', alpha = 0.7, label='beta')
        ax.set_ylabel('regressed slope (pt/sec)')
        ax.axhline(self.beta_threshold, color='k', linestyle='--', lw=1, alpha = 0.2)
        ax.axhline(-self.beta_threshold, color='k', linestyle='--', lw=1, alpha = 0.2)
        ax.fill_between(self.betaSeries.index, self.beta_threshold, self.betaSeries.second,  where= self.betaSeries.second >= self.beta_threshold, alpha = 0.2)
        ax.fill_between(self.betaSeries.index, -self.beta_threshold, self.betaSeries.second, where= self.betaSeries.second <= - self.beta_threshold, alpha = 0.2)
