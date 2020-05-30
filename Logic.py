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