import pandas as pd
import os, shutil
import glob

csv = glob.glob("*.csv")[0]
df = pd.read_csv(csv)
# df = pd.read_csv('test.csv')
print('Finished loading in database ...')
print(df.shape)
df['DateTime'] = pd.to_datetime(df.DateTime, format = "%Y%m%d %H:%M:%S.%f")
df['date'] = df.DateTime.dt.date
df['tradeTime'] = df.DateTime
df['price'] = df['Bid'] * 0.5 + df['Ask'] * 0.5
df['high'] = df.Ask
df['low'] = df.Bid
grps = df.groupby('date')
try:
    shutil.rmtree('../data')
except:
    pass
os.mkdir('../data')
for name, grp in grps:
    print(name)
    fname = str(name).replace('-', '') + '.csv'
    grp.to_csv(os.path.join('../data', fname), index=None)
