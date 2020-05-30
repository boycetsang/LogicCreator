# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:25:41 2017

@author: Boyce
"""

# A helper function to get all veried parameters into a csv
import LogicCreator
from LogicCreator import defaultSettings
import inspect
import pandas as pd
from collections import OrderedDict

def obtainSettings():
    settings = OrderedDict()
    settings = defaultSettings(settings)
    
    logicClasses = LogicCreator.Logic.__subclasses__()
    for logicClass in logicClasses:
        logicName = logicClass.__name__.replace('Logic','')
        args = inspect.getargspec(logicClass).args
        
        args.remove('self')
        args = [logicName + '_' + arg for arg in args]
        for ii in range(len(args)):
            settings[args[ii]] = inspect.getargspec(logicClass).defaults[ii]
    settingsTable = pd.DataFrame(settings, index = ['default']).T
#    print(settingsTable)
    settingsTable['min'] = settingsTable['default']
    settingsTable['max'] = settingsTable['default']
    settingsTable['no_of_steps'] = 20
    settingsTable['isOpt'] = 0
    print(settingsTable)
    settingsTable.to_csv('defaultValues.csv', index_label = 'setting')
    return settings

if __name__ == '__main__':
    obtainSettings()