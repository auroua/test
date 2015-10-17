__author__ = 'auroua'
# from pymc import Dis,Exponential,deterministic,Poisson,Uniform
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np


disasters_array =  \
    np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                   3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

switchpoint = DiscreteUniform('switchpoint', lower=0, upper=110, doc='Switchpoint[year]')

early_mean = Exponential('early_mean',beta=1.)
late_mean = Exponential('late_mean',beta=1.)

@deterministic(plot=False)
def rate(s=switchpoint,e=early_mean,l=late_mean):
    '''Concatenate Poisson means'''
    out = np.empty(len(disasters_array))
    out[:s] = e
    out[s:] = l
    return out

disasters = Poisson('disasters',mu=rate,value=disasters_array,observed=True)

# # The inefficient way, using the Impute function:
# D = Impute('D', Poisson, disasters_array, mu=r)
#
# The efficient way, using masked arrays:
# Generate masked array. Where the mask is true,
# the value is taken as missing.
masked_values = np.ma.masked_array(disasters_array, mask=disasters_array==-999)

# Pass masked array to data stochastic, and it does the right thing
disasters = Poisson('disasters', mu=rate, value=masked_values, observed=True)



# def switchpoint_logp(value, t_l, t_h):
#     if value > t_h or value < t_l:
#         return -np.inf
#     else:
#         return -np.log(t_h - t_l + 1)
#
# def switchpoint_rand(t_l, t_h):
#     from numpy.random import random
#     return np.round( (t_l - t_h) * random() ) + t_l
#
# switchpoint = Stochastic( logp = switchpoint_logp,
#                 doc = 'The switchpoint for the rate of disaster occurrence.',
#                 name = 'switchpoint',
#                 parents = {'t_l': 1851, 't_h': 1962},
#                 random = switchpoint_rand,
#                 trace = True,
#                 value = 1900,
#                 dtype=int,
#                 rseed = 1.,
#                 observed = False,
#                 cache_depth = 2,
#                 plot=True,
#                 verbose = 0)