#encoding:UTF-8
__author__ = 'auroua'
'''
pandas test
'''

import numpy as np
from pandas import Series,DataFrame

sers = Series([1,2,3,4,5],index=['a','b','c','d','e'])
sers.name = 'helloworld'
print sers
print sers['a']

print np.exp(sers)

sdata = {'a':5000,'b':3000,'c':3000,'d':1091291}
sers2 = Series(sdata)
print sers2

data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],
        'year':[2000,2001,2002,2001,2002],
        'pop':[1.5,1.7,3.6,2.4,2.9]}

frame = DataFrame(data)

print frame
print frame.year
print frame.ix[3]

frame['debt'] = 13

print frame

obj3 = Series(['blue','purple','yellow'],index=[0,2,4])
obj4 = obj3.reindex(range(6),method='ffill')
print obj4

obj5 = DataFrame(np.arange(16).reshape(4,4),
                 index=['Ohio','Colorado','Utah','New York'],
                 columns=['one','two','three','four'])

print obj5
print obj5['one']
print obj5[:2]
obj5[obj5<5]=3
print obj5
print obj5.ix['Ohio',['one','two']]

s1 = Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
s2 = Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])

print s1+s2

df1 = DataFrame(np.arange(9).reshape((3,3)),columns=list('bcd'),index=['Ohin','Texa','Colorado'])
df2 = DataFrame(np.arange(12).reshape((4,3)),columns=list('bcd'),index=['Utah','Ohin','Texa','Colorado'])

print df1+df2
print df1.add(df2,fill_value=0)

series2 = df2.ix[0]

print df2-series2

ff = lambda x:x.max()-x.min()

print df2.apply(ff)
print df2.apply(ff,axis=1)

df3 = DataFrame(np.random.randn(3,3),columns=list('bcd'),index=['Ohin','Texa','Colorado'])
ff2 = lambda x:'%.2f'%x
print df3
print df3.applymap(ff2)
print df3
print df3.sort_index(by='b')