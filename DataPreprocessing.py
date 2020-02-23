import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import os
l=[]
path="./dataset/201*"
print(glob.glob(path))
for filename in glob.glob(path):
    with open(filename) as csvfile:
        csvdata=pd.read_csv(csvfile)
        l.append(csvdata)
combined_csv=pd.concat(l,axis=0,ignore_index=True,sort=False)
combined_csv.to_csv(os.path.splitext("./dataset/CombinedData")[0]+ '_all.csv',index=False,sep=",")
