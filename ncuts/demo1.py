__author__ = 'auroua'
import pymc as pm
''' pymc test program '''

parameter = pm.Exponential('poisson_param',1)
data_generator = pm.Poisson('data_generator',parameter)
data_plus_one = data_generator + 1