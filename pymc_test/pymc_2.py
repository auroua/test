__author__ = 'auroua'
import pymc as pm
import numpy as np

n=5*np.ones(4,dtype=int)
x = np.array([-.86,-.3,-.05,.73])

alpha = pm.Normal('alpha',mu=0,tau=.01)
beta = pm.Normal('beta',mu=0,tau=.01)

@pm.deterministic
def theta(a=alpha,b=beta):
    return pm.invlogit(a+b*x)

d = pm.Binomial('d', n=n, p=theta, value=np.array([0.,1.,3.,5.]),\
                  observed=True)