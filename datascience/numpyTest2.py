__author__ = 'auroua'
import numpy as np
arr = np.arange(6).reshape((3,2))
arr2 = np.random.randn(3,2)

arr = np.random.randn(3,4,5)

depth_means = arr.mean(2)

print depth_means

demeand = arr - depth_means[:,:,np.newaxis]

depth_means1 = arr.mean(1)

print depth_means1

demeand2 = arr - depth_means1[:,np.newaxis,:]

print demeand2.mean(1)

arr3 = np.random.randn(5,5)

print arr3
print '111111111111111111111111111111'
print arr3[::3]

print arr3[:,:-1]
print arr3[:,1:]

print np.logical_and.reduce(arr3[:,:-1]>arr3[:,1:],axis=1)

arr4 = np.arange(15)

print np.add.reduceat(arr4,[4,7,11])