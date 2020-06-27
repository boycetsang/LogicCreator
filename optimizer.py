# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:14:22 2017

@author: Boyce
"""

import array
import random
import readline
import ast
from deap import algorithms

import LogicCreator
import pandas as pd
import os, sys
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import time, math

from LogicCreator import defaultSettings
import inspect
from collections import OrderedDict
import multiprocessing
from shutil import copyfile, copy, copytree
import glob

OUTPUT_FILE = "calibration_result.csv"


def textMenu(strList, title="CHOOSE FROM FOLLOWING"):
    bigSepStr = 30 * "="
    sepStr = 30 * "-"
    print(bigSepStr)
    print(title)
    print(bigSepStr)
    cnt = 1
    for str_element in strList:
        print(str(cnt) + ".", str_element)
        cnt += 1
    print(sepStr)
    choice = input("Enter your choice [1-" + str(cnt - 1) + "] : ")
    if choice != "":
        return int(choice) - 1
    else:
        return -1


def printLogicSettings(logicSettings):
    BOLD = "\033[1m"
    END = "\033[0m"
    for logicClass in logicSettings.keys():
        print("For", logicClass.__name__ + ":")
        if logicSettings[logicClass]["status"]:
            for key in logicSettings[logicClass].keys():
                if key != "status":
                    print("%s: %s" % (key, logicSettings[logicClass][key]))
        else:
            print("## TURNED OFF ##")
            for key in logicSettings[logicClass].keys():
                if key != "status":
                    print("%s: %s" % (key, logicSettings[logicClass][key]))
        print()


def printOptVars(optVars):
    optTable = pd.DataFrame(optVars)
    print(optTable.T)


#    for optVar in optVars.keys():
#        print( 'For', optVar +":" )
#        print(optVars[optVar])
#        print()


def try_convert(s):
    if s is None:
        return None
    if s == "":
        return None
    try:
        if float(s).is_integer():
            return int(s)
        else:
            return float(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            if s.lower() == "true":
                return True
            elif s.lower() == "false":
                return False
            else:
                return s


def settingNamesFromLogicSettings(logicSettings):
    logicSettingNames = []
    for logicClass in logicSettings.keys():
        logicSettingNames += list(logicSettings[logicClass].keys())
    return logicSettingNames


def obtainSettings():
    settings = OrderedDict()
    logicSettings = OrderedDict()
    readInTable = pd.read_csv(
        "optSettings.csv", index_col=0, keep_default_na=False, na_values=["NA"]
    )
    defaultValues = readInTable.default

    optVars = OrderedDict()

    for index, val in defaultValues.iteritems():
        settings[index] = try_convert(val)

    logicClasses = LogicCreator.Logic.__subclasses__()
    logicCollection = []
    logicIndexList = []
    print("Loading Logic classes ...")
    for logicClass in logicClasses:
        logicSettings[logicClass] = OrderedDict({"status": True})
        print("Loading", logicClass.__name__, "...")
        logicName = logicClass.__name__.replace("Logic", "")

        args = inspect.getfullargspec(logicClass).args
        args.remove("self")
        args = [logicName + "_" + arg for arg in args]
        for ii in range(len(args)):
            if args[ii] in defaultValues.index:
                logicSettings[logicClass].update(
                    {args[ii]: try_convert(defaultValues[args[ii]])}
                )
                del settings[args[ii]]
                if not logicClass in logicCollection:
                    logicCollection.append(logicClass)
                    defaultValuesRI = defaultValues.reset_index()
                    logicIndexList.append(
                        defaultValuesRI[
                            defaultValuesRI.setting == args[ii]
                        ].index.tolist()[0]
                    )
            else:
                print(
                    args[ii],
                    "settings for",
                    logicClass.__name__,
                    "is not found, loading its default values and turning the module off ...",
                )
                logicSettings[logicClass]["status"] = False
                #                settings[args[ii]] = inspect.getargspec(logicClass).defaults[ii]
                logicSettings[logicClass].update(
                    {args[ii]: inspect.getargspec(logicClass).defaults[ii]}
                )
    #    pd.DataFrame(settings, index = ['default']).T.to_csv('defaultValues.csv', index_label = 'setting')
    logicCollection = [x for (y, x) in sorted(zip(logicIndexList, logicCollection))]
    logicSettings = OrderedDict(
        (logic, logicSettings[logic]) for logic in logicCollection
    )

    optTable = readInTable[readInTable.isOpt > 0]
    del optTable["isOpt"]
    del optTable["default"]
    #    optVars = OrderedDict(optTable.T.to_dict())
    optVars = OrderedDict(
        [
            (
                row.name,
                OrderedDict(
                    [
                        ("min", row["min"]),
                        ("max", row["max"]),
                        ("no_of_steps", row["no_of_steps"]),
                    ]
                ),
            )
            for i, row in optTable.iterrows()
        ]
    )
    #    for i, row in optTable.iterrows():
    #        print(row)
    print(optVars)
    return settings, logicSettings, optVars


#    outputSettings = {}


from deap import algorithms
from deap import base
from deap import creator
from deap import tools

lock = None
soln_cnt = 0
# You'll need these imports in your own code


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="b", fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 100)


def writeLineToFile(l, writeStr):
    if l:
        l.acquire()
        with open(OUTPUT_FILE, "a") as myfile:
            myfile.write(writeStr + "\n")
        l.release()
        pass


def evalOneMax(individual, settings, logicSettings, optVars):
    global veri
    global soln_cnt

    dataSettings = dict()
    calSettings = dict()
    outputSettings = dict()
    calSettings["slippery"] = settings["slippery"]

    outputSettings["graphOutput"] = settings["graphOutput"]
    outputSettings["summaryOutput"] = settings["summaryOutput"]

    #    strategy.strategySettings = StrategySettings(settings)
    dataSettings["dataPath"] = settings["dataPath"]
    dataSettings["dateStart"] = settings["dateStart"]
    dataSettings["dateEnd"] = settings["dateEnd"]
    dataSettings["getDataDaysBefore"] = settings["dataPath"]
    logicSettingNames = settingNamesFromLogicSettings(logicSettings)
    cnt = 0
    logicSettingsExecute = OrderedDict()
    optVarValue = OrderedDict(optVars)
    #    print(optVars, optVarValue)
    for logicClass in logicSettings.keys():
        if logicSettings[logicClass]["status"]:
            logicSettingsExecute.update({logicClass: logicSettings[logicClass]})

    for optVar in optVars.keys():
        min_val = float(optVars[optVar]["min"])
        max_val = float(optVars[optVar]["max"])
        step_cnt = int(optVars[optVar]["no_of_steps"])
        optVarValue[optVar] = round(
            min_val
            + int(individual[cnt] / 100 * step_cnt) / step_cnt * (max_val - min_val),
            7,
        )
        cnt += 1
        if not optVar in logicSettingNames:
            settings[optVar] = optVarValue[optVar]
        else:
            for logicClass in logicSettings.keys():
                for logicClassParam in logicSettings[logicClass].keys():
                    if optVar == logicClassParam:
                        logicSettings[logicClass][optVar] = optVarValue[optVar]

            if logicSettings[logicClass]["status"]:
                logicSettingsExecute.update({logicClass: logicSettings[logicClass]})

    for logicClass in logicSettingsExecute.keys():
        logicSettingsExecute[logicClass] = {
            k.replace(logicClass.__name__.replace("Logic", "") + "_", ""): v
            for k, v in logicSettings[logicClass].items()
        }
        logicSettingsExecute[logicClass].pop("status")

    #    print(logicSettingsExecute)
    strategy = LogicCreator.Strategy(
        settings=settings,
        dataSettings=dataSettings,
        logicSettings=logicSettingsExecute,
        calSettings=calSettings,
        outputSettings=outputSettings,
    )
    # strategy.evalStrat()

    strategy.evalStrat(silence=not veri)

    #    print(strategy)
    metric = []
    pnl = strategy.strategyOutput.getRealPnL(strategy)
    realPnLPerTradedDay = strategy.strategyOutput.getRealPnLPerTradedDay(strategy)
    realPnLPerDay = strategy.strategyOutput.getRealPnLPerDay(strategy)
    realPnLStdDay = strategy.strategyOutput.getRealPnLStdDay(strategy)
    realPnLStdMonth = strategy.strategyOutput.getRealPnLStdMonth(strategy)
    realPnLPerTrade = strategy.strategyOutput.getRealPnLPerTrade(strategy)
    realPnLMaxDrawBackInDay = strategy.strategyOutput.getRealPnLMaxDrawBackInDay(
        strategy
    )
    realPnLMaxDrawBack = strategy.strategyOutput.getRealPnLMaxDrawBack(strategy)
    realWinDayRatio = strategy.strategyOutput.getRealWinDayRatio(strategy)

    metric.append(str(int(pnl)))
    metric.append(str(realPnLPerTradedDay))
    metric.append(str(realPnLPerDay))
    metric.append(str(realPnLStdDay))
    metric.append(str(realPnLStdMonth))
    metric.append(str(realPnLPerTrade))
    metric.append(str(realPnLMaxDrawBackInDay))
    metric.append(str(realPnLMaxDrawBack))
    metric.append(str(realWinDayRatio))
    fitness = pnl + 0.1 * realPnLMaxDrawBack + realPnLPerTrade * 1000

    global counter
    if lock:
        with counter.get_lock():
            counter.value += 1
        writeStr = ",".join(
            [str(counter.value)]
            + [str(optVarValue[x]) for x in optVarValue.keys()]
            + metric
            + [str(fitness)]
        )
    else:
        writeStr = ",".join(
            [str(soln_cnt)]
            + [str(optVarValue[x]) for x in optVarValue.keys()]
            + metric
            + [str(fitness)]
        )

    print(writeStr, flush=True)
    #    print(optVars)
    #    print(optVarValue)
    writeLineToFile(lock, writeStr)
    soln_cnt += 1
    if veri:
        strategy.strategyOutput.outputToFile(strategy)
        writeStr = ",".join(
            [str(counter.value)]
            + [str(optVarValue[x]) for x in optVarValue.keys()]
            + metric
            + [str(fitness)]
        )
        print(writeStr)
        sys.exit(0)

    return (fitness,)


def main(
    pool=None,
    lock=None,
    optSettings=OrderedDict(
        [
            ("indpb", 0.15),
            ("tournsize", 3),
            ("noOfCore", multiprocessing.cpu_count() - 1),
            ("pop_n", 100),
            ("ngen", 9),
            ("mutpb", 0.1),
            ("cxpb", 0.5),
        ]
    ),
):
    random.seed(64)

    if pool:
        toolbox.register("map", pool.map)

    pop = toolbox.population(n=optSettings["pop_n"])
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=optSettings["cxpb"],
        mutpb=optSettings["mutpb"],
        ngen=optSettings["ngen"],
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    print(hof)
    return pop, log, hof


def init(l, args):
    global lock
    lock = l
    global counter
    counter = args


counter = None
veri = None
if __name__ == "__main__":
    print("Welcome to LogicCreator Genetic Optimizer ... ")
    print("Initializing ...")
    #    importPackages()

    settings, logicSettings, optVars = obtainSettings()
    optSettings = OrderedDict(
        [
            ("noOfCore", multiprocessing.cpu_count() - 1),
            ("indpb", 0.15),
            ("tournsize", 3),
            ("pop_n", 200),
            ("ngen", 90),
            ("mutpb", 0.1),
            ("cxpb", 0.5),
        ]
    )
    #    optSettings['noOfCore'] = 1
    #    optSettings['pop_n'] = 1
    #    optSettings['ngen'] = 2
    print("Finished initialization ... Please review current config ...")
    print("")
    print("=-" * 15)
    #    print(pd.DataFrame(settings, index = ['default']).T)
    #    printLogicSettings(logicSettings)

    go = False
    while not go:
        print("\n\n\n")
        print("=-" * 15)
        print("-=" * 15)
        print("OPTIMIZATION SETTINGS")
        print(pd.DataFrame(optSettings, index=[0]).T.to_string(header=False))
        print("=-" * 15)
        print("GENERAL SETTINGS")
        print(pd.DataFrame(settings, index=["default"]).T)
        print()
        print("=-" * 15)
        print("LOGIC MODULE SETTINGS")
        printLogicSettings(logicSettings)
        print("=-" * 15)
        print("OPTIMIZATION VARIABLE SETTINGS")
        if optVars:
            printOptVars(optVars)
        else:
            print("NONE")
        print("=-" * 15)
        print("-=" * 15)
        print("\n\n\n")

        options = [
            "Change settings",
            "Modify existing optimization variables",
            "Add new optimization variables",
            "Enable/Disable Logic modules",
            "Change settings of Logic modules",
            "Change Optimization settings",
            "Calibration Re-run/Verification Run",
            "Run Program",
        ]
        choice = textMenu(options, title="   OPTIMIZATION CONFIG MENU  ")

        ### Take action as per selected menu-option ###
        if choice == 0:
            print("Pulling up the settings ...")
            settingKeys = list(settings.keys())
            choice = textMenu(settingKeys)
            newVal = input(
                "Change the value of " + str(settingKeys[choice]) + " to ... "
            )
            newVal = try_convert(newVal)
            settings[settingKeys[choice]] = newVal

        elif choice == 1:
            print("Showing optimization variables ...")
            optVarsKeys = list(optVars.keys())
            choice = textMenu(
                optVarsKeys, title="CHOOSE OPTIMIZATION VARIABLE TO CHANGE"
            )
            optVarKey = optVarsKeys[choice]
            optVarSettings = optVars[optVarKey]
            optVarSettingsKeys = list(optVarSettings.keys())
            optVarSettingsKeysShow = [
                optVarSettingsKey + ": " + str(optVarSettings[optVarSettingsKey])
                for optVarSettingsKey in optVarSettingsKeys
            ]
            optVarSettingsKeys += ["Remove from optimization"]
            optVarSettingsKeysShow += ["Remove from optimization"]
            choice = textMenu(optVarSettingsKeys, title="CHOOSE SETTINGS TO MODIFY")
            if choice == len(optVarSettingsKeys) - 1:
                del optVars[optVarKey]
            else:
                newVal = input(
                    "Change the value of "
                    + str(optVarSettingsKeys[choice])
                    + " to ... "
                )
                newVal = try_convert(newVal)
                optVarSettings[optVarSettingsKeys[choice]] = newVal

        elif choice == 2:
            choice = textMenu(
                ["Add general setting type", "Add logic setting type"],
                title="CHOOSE OPTIMIZATION VARIABLE TO CHANGE",
            )
            if choice == 0:
                #                print ("Showing non-optimized variables ...")
                settingKeys = list(settings.keys())
                choice = textMenu(settingKeys)
                if not settingKeys[choice] in list(optVars.keys()):
                    print("Creating new optimization variable on", settingKeys[choice])
                    newVal = input(
                        "Min value of " + str(settingKeys[choice]) + " is ... "
                    )
                    min_val = try_convert(newVal)
                    newVal = input(
                        "Max value of " + str(settingKeys[choice]) + " is ... "
                    )
                    max_val = try_convert(newVal)
                    newVal = input(
                        "Step count of " + str(settingKeys[choice]) + " is ... "
                    )
                    step_cnt = int(newVal)
                    optVars[settingKeys[choice]] = {
                        "min": min_val,
                        "max": max_val,
                        "no_of_steps": step_cnt,
                    }
                else:
                    print(
                        settingKeys[choice]
                        + " is already in the opitmization varaible set!"
                    )
            else:
                logicClasses = [logicClass for logicClass in logicSettings.keys()]
                logicClassesShow = [
                    logicClass.__name__
                    + " Current status :\t"
                    + str(logicSettings[logicClass]["status"])
                    for logicClass in logicSettings.keys()
                ]
                choice = textMenu(logicClassesShow, title="CHOOSE LOGIC MODULES TO ADD")
                logicClass = logicClasses[choice]
                settingKeys = list(logicSettings[logicClass].keys())
                settingKeys.remove("status")
                choice = textMenu(settingKeys)
                if not settingKeys[choice] in list(optVars.keys()):
                    print("Creating new optimization variable on", settingKeys[choice])
                    newVal = input(
                        "Min value of " + str(settingKeys[choice]) + " is ... "
                    )
                    min_val = try_convert(newVal)
                    newVal = input(
                        "Max value of " + str(settingKeys[choice]) + " is ... "
                    )
                    max_val = try_convert(newVal)
                    newVal = input(
                        "Step count of " + str(settingKeys[choice]) + " is ... "
                    )
                    step_cnt = int(newVal)
                    optVars[settingKeys[choice]] = {
                        "min": min_val,
                        "max": max_val,
                        "no_of_steps": step_cnt,
                    }
                else:
                    print(
                        settingKeys[choice]
                        + " is already in the opitmization varaible set!"
                    )

        elif choice == 3:
            print("Locating Logic modules...")
            logicClasses = [logicClass for logicClass in logicSettings.keys()]
            logicClassesShow = [
                logicClass.__name__
                + " Current status :\t"
                + str(logicSettings[logicClass]["status"])
                for logicClass in logicSettings.keys()
            ]
            choice = textMenu(logicClassesShow, title="CHOOSE LOGIC MODULES TO TOGGLE")
            if choice != -1:
                logicSettings[logicClasses[choice]]["status"] = not logicSettings[
                    logicClasses[choice]
                ]["status"]
                if logicSettings[logicClasses[choice]]["status"]:
                    tmp = logicSettings[logicClasses[choice]]
                    logicSettings.pop(logicClasses[choice])
                    logicSettings[logicClasses[choice]] = tmp

        elif choice == 4:
            print("Locating Logic modules...")
            logicClasses = [logicClass for logicClass in logicSettings.keys()]
            logicClassesShow = [
                logicClass.__name__
                + " Current status :\t"
                + str(logicSettings[logicClass]["status"])
                for logicClass in logicSettings.keys()
            ]
            choice = textMenu(logicClassesShow, title="CHOOSE LOGIC MODULES TO TOGGLE")
            logicClass = logicClasses[choice]
            settingKeys = list(logicSettings[logicClass].keys())
            settingKeys.remove("status")
            choice = textMenu(settingKeys)
            newVal = input(
                "Change the value of " + str(settingKeys[choice]) + " to ... "
            )
            newVal = try_convert(newVal)

            logicSettings[logicClass][settingKeys[choice]] = newVal
        elif choice == 5:
            print("Pulling up optimization settings")
            optKeys = list(optSettings.keys())
            choice = textMenu(optKeys, title="CHOOSE OPTIMIZATION PARAMETERS TO MODIFY")
            newVal = input("Change the value of " + str(optKeys[choice]) + " to ... ")
            newVal = try_convert(newVal)
            optSettings[optKeys[choice]] = newVal
        elif choice == 6:
            settings = dict()
            dataSettings = dict()
            logicSettings = OrderedDict()
            calSettings = dict()
            outputSettings = dict()
            run_dir = glob.glob("EXPT_RUN*")

            print("Available calibration results ...  choose from following")

            print("\n".join(run_dir))

            run_dir_chosen = input("Enter the calibration set number \n")
            run_dir_chosen = "EXPT_RUN_" + run_dir_chosen
            os.chdir(run_dir_chosen)

            result_csv = pd.read_csv("calibration_result.csv")
            opt_csv = pd.read_csv("optSettings.csv", index_col="setting")

            settings = OrderedDict((ind, opt_csv.default[ind]) for ind in opt_csv.index)
            max_col = pd.get_option("display.width")
            max_row = pd.get_option("max_row")
            try:
                pd.set_option("display.width", 1000)
                pd.set_option("max_row", 1000)
                print(result_csv.sort_values(by="fitness", ascending=False))
                pd.set_option("display.width", max_col)
                pd.set_option("max_row", max_row)
            except:
                print("Printing calibration_result.csv failed...")
            solnid = int(input("Input solutionID: "))
            result_csv = result_csv[result_csv.SolutionID == solnid]
            mdl_dir = "MDL_" + str(solnid)
            if not os.path.isdir(mdl_dir):
                os.mkdir(mdl_dir)

            for col in result_csv.columns.values:
                if col in settings.keys():
                    settings[col] = result_csv[col].iloc[0]
                    opt_csv.default[col] = settings[col]

            for setting in settings:
                if isinstance(settings[setting], float):
                    if np.isnan(settings[setting]):
                        settings[setting] = None
            print(
                "This run was carried out from "
                + settings["dateStart"]
                + " to "
                + settings["dateEnd"]
            )
            choice = textMenu(
                [
                    "Calibration Re-run (Parallel)",
                    "Calibration Re-run",
                    "Verification Run (Parallel)",
                    "Verification Run",
                    "Calibration and Verification Run (Parallel)",
                ]
            )
            outputSettings["graphOutput"] = settings["graphOutput"]

            doCaliToo = True
            while doCaliToo:
                doCaliToo = False
                if not os.path.exists(mdl_dir):
                    os.mkdir(mdl_dir)
                opt_csv.to_csv(os.path.join(
                        mdl_dir, 'selected_model.csv'))
                if choice == 0:
                    outputSettings["summaryOutput"] = os.path.join(
                        mdl_dir, "Summary_calibration.csv"
                    )

                    dataSettings["dateStart"] = settings["dateStart"]
                    dataSettings["dateEnd"] = settings["dateEnd"]
                if choice == 1:
                    outputSettings["summaryOutput"] = os.path.join(
                        mdl_dir, "Summary_calibration.csv"
                    )

                    dataSettings["dateStart"] = settings["dateStart"]
                    dataSettings["dateEnd"] = settings["dateEnd"]
                if choice == 2 or choice == 4:

                    outputSettings["summaryOutput"] = os.path.join(
                        mdl_dir, "Summary_verification.csv"
                    )

                    dataSettings["dateStart"] = input("Input new dateStart ... ")
                    dataSettings["dateEnd"] = input("Input new dateEnd ... ")
                    if choice == 4:
                        choice = 2
                        doCaliToo = True
                if choice == 3:
                    outputSettings["summaryOutput"] = os.path.join(
                        mdl_dir, "Summary_verification.csv"
                    )

                    dataSettings["dateStart"] = input("Input new dateStart ... ")
                    dataSettings["dateEnd"] = input("Input new dateEnd ... ")

                dataSettings["dataPath"] = settings["dataPath"]
                calSettings["slippery"] = float(settings["slippery"])

                logicClasses = LogicCreator.Logic.__subclasses__()
                for setting in settings.keys():
                    appearedSettings = {}
                    for logicClass in logicClasses:
                        logicName = logicClass.__name__.replace("Logic", "")
                        if setting.startswith(logicName):
                            appearedSettings[
                                setting.replace(logicName + "_", "")
                            ] = try_convert(settings[setting])
                            if logicClass in logicSettings.keys():
                                logicSettings[logicClass].update(appearedSettings)
                            else:
                                logicSettings[logicClass] = appearedSettings

                strategy = LogicCreator.Strategy(
                    settings=settings,
                    dataSettings=dataSettings,
                    logicSettings=logicSettings,
                    calSettings=calSettings,
                    outputSettings=outputSettings,
                )
                if choice == 1 or choice == 3:
                    strategy.evalStrat()
                else:
                    strategy.evalStratParallel()
                print(strategy.strategyOutput.outputTable)
                strategy.strategyOutput.outputToFile(strategy)
                realPnL = strategy.strategyOutput.getRealPnL(strategy)
                print("Real PnL for the strategy is: ", realPnL)

                metric = []
                pnl = strategy.strategyOutput.getRealPnL(strategy)

                realPnLPerTradedDay = strategy.strategyOutput.getRealPnLPerTradedDay(
                    strategy
                )
                realPnLPerDay = strategy.strategyOutput.getRealPnLPerDay(strategy)
                realPnLStdDay = strategy.strategyOutput.getRealPnLStdDay(strategy)
                realPnLStdMonth = strategy.strategyOutput.getRealPnLStdMonth(strategy)
                realPnLPerTrade = strategy.strategyOutput.getRealPnLPerTrade(strategy)
                realPnLMaxDrawBackInDay = strategy.strategyOutput.getRealPnLMaxDrawBackInDay(
                    strategy
                )
                realPnLMaxDrawBack = strategy.strategyOutput.getRealPnLMaxDrawBack(
                    strategy
                )
                realWinDayRatio = strategy.strategyOutput.getRealWinDayRatio(strategy)

                performance_dict = {
                    "realPnLPerTradedDay": realPnLPerTradedDay,
                    "realPnLPerDay": realPnLPerDay,
                    "realPnLStdDay": realPnLStdDay,
                    "realPnLStdMonth": realPnLStdMonth,
                    "realPnLPerTrade": realPnLPerTrade,
                    "realPnLMaxDrawBackInDay": realPnLMaxDrawBackInDay,
                    "realPnLMaxDrawBack": realPnLMaxDrawBack,
                    "realWinDayRatio": realWinDayRatio,
                }
                performance_df = pd.DataFrame(performance_dict, index=[0])

                metric.append(str(int(pnl)))
                metric.append(str(realPnLPerTradedDay))
                metric.append(str(realPnLPerDay))
                metric.append(str(realPnLStdDay))
                metric.append(str(realPnLStdMonth))
                metric.append(str(realPnLPerTrade))
                metric.append(str(realPnLMaxDrawBackInDay))
                metric.append(str(realPnLMaxDrawBack))
                metric.append(str(realWinDayRatio))
                print(
                    "\t".join(
                        [
                            "TotalRealPnL",
                            "RealPnLPerTradedDay",
                            "RealPnLPerDay",
                            "RealPnLStdDay",
                            "RealPnLStdMonth",
                            "PnLPerTrade",
                            "RealPnLMaxDrawBackInDay",
                            "RealPnLMaxDrawBack",
                            "RealWinDayRatio",
                        ]
                    )
                    + "\n"
                )

                print("\t".join(metric))
                if choice == 0 or choice == 1:
                    performance_df.to_csv(
                        os.path.join(mdl_dir, "Performance_calibration.csv")
                    )
                    strategy.strategyOutput.outputGraphs(strategy)
                else:
                    performance_df.to_csv(
                        os.path.join(mdl_dir, "Performance_verification.csv")
                    )
                if doCaliToo:
                    choice = 0

            input("Evaluation Finished ... press any key to exit")
            exit()
        elif choice == 7:
            go = True
            print("Using Logic modules ...")
            print(
                "## CAUTION: The order of Logic modules is important due to possible combination of and/or conditions! ##"
            )
            print(
                "To change the order of the Logic modules, go back or change the order in optSettings.csv"
            )
            print()
            print("-=-" * 15)
            print(
                " > ".join(
                    [
                        logicClass.__name__
                        for logicClass in logicSettings.keys()
                        if logicSettings[logicClass]["status"]
                    ]
                )
            )
            print("-=-" * 15)
            print()
            test_gold_dir = "EXPT_RUN"
            cwd = os.getcwd()
            i = 0
            while i < 100:
                new_test_dir = os.path.join(cwd, test_gold_dir + "_" + str(i))
                if not os.path.exists(new_test_dir):
                    os.mkdir(new_test_dir)
                    os.chdir(new_test_dir)
                    break
                elif os.listdir(new_test_dir) == []:
                    os.chdir(new_test_dir)
                    break
                else:
                    i += 1

            if optVars:
                printOptVars(optVars)
                choice = input(
                    "Here is the optimization variable setting ... Are you sure to continue? [(Y)/N]"
                )
                if choice.lower() in "no":
                    go = False
                    os.chdir("..")
                veri = False

            else:
                print(
                    "No Optimization variable! This is a verification job. Summary output will be written. 1 CPU will be used"
                )
                go = True
                veri = True
                optSettings["noOfCore"] = 1

            print("Saving current settings to " + os.getcwd())
            df = pd.DataFrame(settings, index=["default"])
            for logic in logicSettings:
                # print(logicSettings[logic])
                if logicSettings[logic]["status"] == True:
                    df_temp = pd.DataFrame(logicSettings[logic], index=["default"])
                    del df_temp["status"]

                    df = pd.concat([df, df_temp], axis=1)
            df = df.T
            df["min"] = df.default
            df["max"] = df.default
            df["no_of_steps"] = 20
            df["isOpt"] = 0

            for optVarKey in optVars.keys():
                currentOptVarProp = optVars[optVarKey]
                if optVarKey in df.index:
                    df.loc[optVarKey, "min"] = currentOptVarProp["min"]
                    df.loc[optVarKey, "max"] = currentOptVarProp["max"]
                    df.loc[optVarKey, "no_of_steps"] = currentOptVarProp["no_of_steps"]
                    df.loc[optVarKey, "isOpt"] = 1
            df.to_csv("optSettings.csv", na_rep="", index_label="setting")

        else:  ## default: reprint menu ##
            pass
    print("")

    toolbox.register(
        "evaluate",
        evalOneMax,
        settings=settings,
        logicSettings=logicSettings,
        optVars=optVars,
    )
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        len(optVars.keys()),
    )
    #    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=optSettings["indpb"])
    toolbox.register("select", tools.selTournament, tournsize=optSettings["tournsize"])
    # Structure initializrs
    header = (
        ",".join(
            ["SolutionID"]
            + list(optVars.keys())
            + [
                "TotalRealPnL",
                "RealPnLPerTradedDay",
                "RealPnLPerDay",
                "RealPnLStdDay",
                "RealPnLStdMonth",
                "PnLPerTrade",
                "RealPnLMaxDrawBackInDay",
                "RealPnLMaxDrawBack",
                "RealWinDayRatio",
                "fitness",
            ]
        )
        + "\n"
    )
    print(header)
    with open(OUTPUT_FILE, "w") as myfile:
        myfile.write(header)
    pool = None

    l = multiprocessing.Lock()
    #    init()
    if optSettings["noOfCore"] > 1:
        counter = multiprocessing.Value("i", 0)
        pool = multiprocessing.Pool(
            optSettings["noOfCore"], initializer=init, initargs=(l, counter)
        )

    main(pool=pool, optSettings=optSettings)
    input("Optimization program finished ... hit any key to exit ...")
