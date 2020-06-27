# LogicCreator
This is a Python-based simple day trading backtester that allow users to customize their entrance/exit long/short logics based on the movement of the same day. It also comes with an optimizer to fine-tune logic parameters. This works only under Linux environments, but today Linux subsystem ([WSL](https://docs.microsoft.com/en-us/learn/modules/get-started-with-windows-subsystem-for-linux/)) is very accessible in windows and this is also where I developed and tested the code.

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
