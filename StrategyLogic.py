class StrategyLogic:
    def __init__(self, logicSettings):
        self.logicSettings = logicSettings
        self.logicList = []
        self.dataNeededDaysBefore = []
        self.logicCollection = []
        self.extraTimeSeries = {}
        for logic in logicSettings.keys():
            logicSetting = logicSettings[logic]
            self.dataNeededDaysBefore += logicSetting.pop("dataNeededDaysBefore", [])
            self.logicCollection += [logic(**logicSetting)]

    def __str__(self):
        return "StrategyLogic: " + str(self.logicSettings) + str(self.logicList)

    def getBuyEntranceCondition(self, strategy, ind):
        buyEntranceCondition = True
        for logicSetting in self.logicCollection:
            if logicSetting.BUY_ENTRANCE_LOGIC_OPERATION == "or":
                buyEntranceCondition = (
                    buyEntranceCondition
                    or logicSetting.addBuyEntranceCondition(strategy, ind)
                )
            else:
                buyEntranceCondition = (
                    buyEntranceCondition
                    and logicSetting.addBuyEntranceCondition(strategy, ind)
                )
            if not buyEntranceCondition:
                return buyEntranceCondition
        return buyEntranceCondition

    def getBuyExitCondition(self, strategy, position, ind):
        buyExitCondition = True
        for logicSetting in self.logicCollection:
            if logicSetting.BUY_EXIT_LOGIC_OPERATION == "or":
                buyExitCondition = buyExitCondition or logicSetting.addBuyExitCondition(
                    strategy, position, ind
                )
            else:
                buyExitCondition = (
                    buyExitCondition
                    and logicSetting.addBuyExitCondition(strategy, position, ind)
                )
        return buyExitCondition

    def getSellEntranceCondition(self, strategy, ind):
        sellEntranceCondition = True
        for logicSetting in self.logicCollection:
            if logicSetting.SELL_ENTRANCE_LOGIC_OPERATION == "or":
                sellEntranceCondition = (
                    sellEntranceCondition
                    or logicSetting.addSellEntranceCondition(strategy, ind)
                )
            else:
                sellEntranceCondition = (
                    sellEntranceCondition
                    and logicSetting.addSellEntranceCondition(strategy, ind)
                )
        return sellEntranceCondition

    def getSellExitCondition(self, strategy, position, ind):
        sellExitCondition = True
        for logicSetting in self.logicCollection:

            if logicSetting.SELL_EXIT_LOGIC_OPERATION == "or":
                logicSetting.addSellExitCondition(strategy, position, ind)
                sellExitCondition = (
                    sellExitCondition
                    or logicSetting.addSellExitCondition(strategy, position, ind)
                )
            else:
                sellExitCondition = (
                    sellExitCondition
                    and logicSetting.addSellExitCondition(strategy, position, ind)
                )

        return sellExitCondition

    def performComputeForDay(self, strategy, timeSeriesTick, timeSeriesTrade):
        for logicSetting in self.logicCollection:
            self.extraTimeSeries.update(
                logicSetting.computeForDay(strategy, timeSeriesTick, timeSeriesTrade)
            )

    def performComputeForAction(self, strategy, ind, val):
        for logicSetting in self.logicCollection:
            logicSetting.computeForAction(strategy, ind, val)

    def getEntranceReport(self, strategy, position):
        for logicSetting in self.logicCollection:
            position = logicSetting.addEntranceReport(strategy, position)

    def getExitReport(self, strategy, position):
        for logicSetting in self.logicCollection:
            position = logicSetting.addExitReport(strategy, position)
