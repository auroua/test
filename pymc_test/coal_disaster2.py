import coal_disaster
from pymc import MCMC
from pylab import hist, show
from pymc.Matplot import plot,pyplot

__author__ = 'auroua'

M = MCMC(coal_disaster)
print M.switchpoint.value

M.sample(iter=10000, burn=1000, thin=10)
# print len(M.trace('switchpoint')[:])
# hist(M.trace('late_mean')[:])
# show()
plot(M)
M.stats()