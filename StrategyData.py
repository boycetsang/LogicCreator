import pandas as pd
import os, sys


class StrategyData:
    def __init__(self, dataSettings, dataNeededDaysBefore=[]):
        self.dataSettings = dataSettings
        self.dateFileDict = {}
        self.currentDate = None
        self.dateStart = pd.to_datetime(
            dataSettings["dateStart"], format="%Y%m%d"
        ).date()
        self.dateEnd = pd.to_datetime(dataSettings["dateEnd"], format="%Y%m%d").date()
        self.tradeDateRange = pd.date_range(self.dateStart, self.dateEnd, freq="D")

        for date in self.tradeDateRange:
            pathName = os.path.join(
                dataSettings["dataPath"], date.strftime("%Y%m%d") + ".csv"
            )
            if os.path.isfile(pathName):
                self.dateFileDict[date.date()] = pathName

            for day in dataNeededDaysBefore:
                pathName = os.path.join(
                    dataSettings["dataPath"],
                    (date - timedelta(days=day)).strftime("%Y%m%d") + ".csv",
                )
                if os.path.isfile(pathName):
                    self.dateFileDict[(date - timedelta(days=day)).date()] = pathName

    def __str__(self):
        return "StrategyData: " + str(self.dataSettings)
