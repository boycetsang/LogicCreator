# LogicCreator
This is a Python-based simple day trading backtester that allow users to customize their entrance/exit long/short logics based on the movement of the same day. It also comes with an optimizer to fine-tune logic parameters. 
* This works only under Linux environments, but today Linux subsystem ([WSL](https://docs.microsoft.com/en-us/learn/modules/get-started-with-windows-subsystem-for-linux/)) is very accessible in Windows. This is also where I developed and tested the code.

## Getting Started
### Setting up Repo and Python virtual environment
Let's get started by preparing your own Python virtual environment. Please install Python 3.6+ on your OS. Then use

`python3 -m venv .venv`

to initialize your python environment.

Next, clone this git repo by 

`git clone https://github.com/boycetsang/LogicCreator.git`

Then, activate the environment by

`source .venv/bin/activate`

install all the required packages by 

`pip install -r requirements.txt`

### Downloading the required backtesting data
We need an external source to download tick-by-tick histoical data and I am not aware of an API to do it automatically.

Instead, we can use this application called [Quant Data Manager](https://strategyquant.com/quantdatamanager/) to acquire historical data.

Pick your favourite stock ticker and download all its histroical data, export to csv and place it under the `data_pre` location
![QuantDataManager](https://github.com/boycetsang/LogicCreator/blob/master/docs/quantapp.JPG)

Before we run the backtester, we need clean and separate the long csv file by date for efficiency.

`cd data_pre; python preprocess.py; cd ..`

### Making sure that everything works
Now everything should be set. Test the backtester by

`python LogicCreator.py`

After the program has finished, you should see a new file `Summary.csv` and a collection of graphs at `graph_output`

You can review them to see how the default logic have worked. It may have made no trades at all.

## Create your own logic
### Customize logic flow
Open up LogicCreator.py with a text editor. Now you have full access to the backtesting settings.

Towards the end of the file, the function `defaultSettings` define all the module-level parameters, such as backtesting date range, output file paths, etc.

In the main section, you can add/remove logic as you wish using a dictionary to define all the logic parameters. You can open up `Logic.py` to see what parameters are available on each logic module.

After changing this file, you can re-run and see how the trades are behaving differently.

### How to use the `Logic` object 
`LogicCreator` is written to allow flexible implementation of time series manipulation using pandas. All trigger (or prohibition) of position entrance are wrapped under the `Logic` class in `Logic.py`. All `Logic` class definition must contain the following functions/attributes (even if they do nothing). You should use the base class as a template and use the existing logic module as examples.

#### BUY_ENTRANCE_LOGIC_OPERATION, SELL_ENTRANCE_LOGIC_OPERATION, BUY_EXIT_LOGIC_OPERATION, SELL_EXIT_LOGIC_OPERATION
When more than 1 logic modules are enabled, the buy/sell entrance/exit logics has to be combined. The order is defined in the order it was added in `LogicCreator.py`. These attributes define the logic left-operation relative to this module. The only possible values are 'and' and 'or'.

#### __init__
The constructor has to take in the logic parameters and assign them to the object.

#### addEntranceReport, addExitReport
When an entrance/exit is triggered, this function is called to allow reporting values to `Summary.csv`

#### addBuyEntranceCondition, addBuyExitCondition, addSellEntranceCondition, addSellExitCondition
For every `interval`, these functions are called to see if a position should be opened/closed.

#### computeForDay
If needed, this function can be used to calculate quantities that are used throughout the day for faster backtesting.

#### computeForAction
This function is called when a position is opened/closed, but not due to this logic.

#### printOnSecondAxis
Only the last logic module that has this function will be effective. This function allow a secondary axis plot on the graph output. Here is an example  
<img src="https://github.com/boycetsang/LogicCreator/blob/master/docs/graph_example.png" width="500">

## Searching for strategies with optimal logic parameters
Instead of manually searching for a best strategy, you can also use the python package `deap` to search for a few optimal solutions and inspect only those solution.

To use the optimizer, initialize the setting files by

`python init_opt.py`

You should see a new csv file called `optSettings.csv`, open it up in excel/text editor. You can then change the default, min, max, and step value for all parameters.

Then, execute the optimizer by 

`python optimizer.py`

You can then change a few things before launching the optimizer. You will see backtesting runs as extra directory in the current location. 

## Cross-validating optimized strategy
THe optimizer can almost always find good in-sample solution - that's why it is a good idea to use a different dataset (e.g. different date range) to challenge the selected strategy. This can be done by the same script, but using the verification option in command line menu.
