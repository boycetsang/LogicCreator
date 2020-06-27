# LogicCreator
This is a Python-based simple day trading backtester that allow users to customize their entrance/exit long/short logics based on the movement of the same day. It also comes with an optimizer to fine-tune logic parameters. 
* This works only under Linux environments, but today Linux subsystem ([WSL](https://docs.microsoft.com/en-us/learn/modules/get-started-with-windows-subsystem-for-linux/)) is very accessible in Windows. This is also where I developed and tested the code.

## Getting Started (10 mins)
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
![QuantDataManager](https://github.com/boycetsang/LogicCreator/blob/master/quantapp.JPG)

Before we run the backtester, we need clean and separate the long csv file by date for efficiency.

`cd data_pre; python preprocess.py; cd ..`

### Making sure that everything works
Now everything should be set. Test the backtester by

`python LogicCreator.py`

After the program has finished, you should see a new file `Summary.csv` and a collection of graphs at `graph_output`

You can review them to see how the default logic have worked. It may have made no trades at all.

### Manually tune logic parameters
Open up LogicCreator.py with a text editor. Now you have full access to the backtesting settings.

Towards the end of the file, the function `defaultSettings` define all the module-level parameters, such as backtesting date range, output file paths, etc.

In the main section, you can add/remove logic as you wish using a dictionary to define all the logic parameters.

After changing this file, you can re-run and see how the trades are behaving differently.
