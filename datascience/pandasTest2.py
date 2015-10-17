#encoding:UTF-8
__author__ = 'auroua'
from pandas import Series,DataFrame
from numpy import nan as NA
import numpy as np
df = DataFrame({'a':np.arange(7),'b':np.arange(7,0,-1),'c':['one',NA,'one','two','two','two','two'],'d':[0,NA,2,3,NA,1,2]})

print df

print df.sum()
print df.sum(axis=1)

print df.describe()

new_df = df.drop('c',axis=1)
new_df = new_df.dropna()
print new_df.corr()
