# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:25:41 2017

@author: Boyce
"""

# A helper function to get all veried parameters into a csv
import os
import LogicCreator
from LogicCreator import defaultSettings
import inspect
import pandas as pd
from collections import OrderedDict

OUTPUT_FILE = "optSettings.csv"


def obtainSettings():
    print("Initializing {} with default values ...".format(OUTPUT_FILE))
    settings = OrderedDict()
    settings = defaultSettings(settings)

    logicClasses = LogicCreator.Logic.__subclasses__()
    for logicClass in logicClasses:
        logicName = logicClass.__name__.replace("Logic", "")
        args = inspect.getargspec(logicClass).args

        args.remove("self")
        args = [logicName + "_" + arg for arg in args]
        for ii in range(len(args)):
            settings[args[ii]] = inspect.getargspec(logicClass).defaults[ii]
    settingsTable = pd.DataFrame(settings, index=["default"]).T

    settingsTable["min"] = settingsTable["default"]
    settingsTable["max"] = settingsTable["default"]
    settingsTable["no_of_steps"] = 20
    settingsTable["isOpt"] = 0
    print(settingsTable)
    if os.path.exists(OUTPUT_FILE):
        choice = input(
            "{} exists ... Are you sure to overwrite? [(Y)/N]".format(OUTPUT_FILE)
        )
        if choice.lower() in "no":
            print("Program aborted.")
            return
    settingsTable.to_csv(OUTPUT_FILE, index_label="setting")
    return


if __name__ == "__main__":
    obtainSettings()
