__author__ = 'auroua'
import pymc
import pymc_2

S = pymc.MCMC(pymc_2, db='pickle')
S.sample(iter=10000, burn=5000, thin=2)
pymc.Matplot.plot(S)