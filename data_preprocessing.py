import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import glob
import os

data=[]
path="./dataset/201*"

for filename in glob.glob(path):
    with open(filename) as csvfile:
        csvdata=pd.read_csv(csvfile)
        data.append(csvdata)

combined_csv=pd.concat(data,axis=0,ignore_index=True,sort=False)
combined_csv.to_csv(os.path.splitext("./dataset/combined_data")[0]+ '_all.csv',index=False,sep=",")

y = combined_csv['happiness_score']
x = combined_csv[['gdp','life_expectancy','freedom','generosity','corruption']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

train = pd.concat([y_train, x_train], axis=1, sort=False)
test = pd.concat([y_test, x_test], axis=1, sort=False)

train.to_csv(os.path.splitext("./dataset/train")[0]+ '.csv',index=False,sep=",")
test.to_csv(os.path.splitext("./dataset/test")[0]+ '.csv',index=False,sep=",")