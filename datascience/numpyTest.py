#encoding:UTF-8
__author__ = 'auroua'

import numpy as np
import matplotlib.pyplot as plt

t1 = np.ones((3,3))

t2 = np.random.randn(3,3)

tt = np.arange(5)
tt2 = tt.reshape((5,1))
tt2[3] = 10
print tt
print tt2
# print t1.dtype
# print t1.shape
#
# print t1
# print t2
# print t2[1]
# t3 =  t2[1,:1]
# print t3.shape
# t4 = t2[:1,:1]
# print t4.shape

# t5 = np.arange(16).reshape((2,2,4))
# print t5.transpose((1,0,2))
#
# points = np.arange(-5,5,0.01)
#
# xs,ys = np.meshgrid(points,points)
#
# result = np.sqrt(xs**2+ys**2)
# print result
# print result.shape
# plt.imshow(result,cmap=plt.cm.gray);plt.colorbar()
# plt.title('Image')
# #plt.show()
#
# xarr = np.array([1.1,1.2,1.3,1.4,1.5])
# yarr = np.array([2.1,2.2,2.3,2.4,2.5])
# cond = np.array([True,False,True,True,False])
#
# print [(y if x else z)for x,y,z in zip(cond,xarr,yarr)]
#
# print np.where(cond,xarr,yarr)
#
# arr = np.random.randn(4,4)
#
# result =  np.where(arr>0,2,-2)
# print arr
# print result
# print arr.mean()
#
# arr2 = np.random.randn(1000)
# arr2.sort()
# print arr2[int(0.05*len(arr2))]
#
# arr3 = np.random.rand(3,3)
# results = arr3.T.dot(arr3)
# print results
#
# from numpy.linalg import inv,qr
# print inv(results)
#
# resultss = np.dot(results,inv(results))
# print resultss
# print resultss.astype('int64')
#
# q,r = qr(results)
#
# print q
# print r

samples = np.random.normal(size=(4,4))
print samples

#random walk
import random
position = 0
walk = []
for i in range(1000):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)

print walk

draws = np.random.randint(0,2,size=1000)
steps = np.where(draws>0,1,-1)
walk = steps.cumsum()
print walk

print "======================================================"
step=10
instances = 10
draws = np.random.randint(0,2,size=(instances,step))
walk = np.where(draws>0,1,-1)
#print walk
walks = walk.cumsum(1)
print walks

hist3 = (np.abs(walks)>=3).any(1)
print hist3

crossing_time = np.abs(walks[hist3]).argmax(1)
crossing_time2 = (np.abs(walks[hist3])>=3).argmax(1)
print crossing_time
print crossing_time2

print  crossing_time.mean()
print  crossing_time2.mean()

import theano_test
from theano_test import tensor
print theano_test.__version__

a = tensor.dscalar()
b = tensor.dscalar()

c = a + b

f = theano_test.function([a,b],c)

print f
assert 4.0 == f(1.5,2.5)