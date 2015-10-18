__author__ = 'auroua'
import numpy as np

# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)

def test_2():
    n=4
    m=5
    a = np.arange(1,n*m+1).reshape(n,m)
    print(a)
    # [[ 1  2  3  4  5]
    #  [ 6  7  8  9 10]
    #  [11 12 13 14 15]
    #  [16 17 18 19 20]]
    sz = a.itemsize
    h,w = a.shape
    bh,bw = 2,2
    shape = (h/bh, w/bw, bh, bw)
    print(shape)
    # (2, 2, 2, 2)

    strides = sz*np.array([w*bh,bw,w,1])
    print(strides)
    # [40  8 20  4]

    blocks=np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    print(blocks)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def moving_average(a,n=3):
    '''caculate the sliding average'''
    ret = np.cumsum(a,dtype=np.float)
    ret[n:] = ret[n:]-ret[:-n]
    return ret[n-1:]/n

def test_equals():
    Z = np.random.randint(0,5,(10,3))
    E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
    U = Z[~E]
    print(Z)
    print(U)
    #my method
    np.all(Z==Z[:,:1].repeat(3,axis=1))


if __name__=='__main__':
    z = rolling(np.arange(10), 3)
    print(z)

    x=np.arange(10).reshape((2,5))
    rolling_window(x, 3)