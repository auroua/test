__author__ = 'auroua'
from thinkbayes import Pmf

# bascal example
# pmf = Pmf()
# for x in range(0, 6):
#     pmf.Set(x, 1/6.0)
#
# pmf.Normalize()
#
# for i in range(0,6):
#     print pmf.Prob(i)

pmf = Pmf()
pmf.Set('Bow1', 0.5)
pmf.Set('Bow2', 0.5)

pmf.Mult('Bow1', 0,75)
pmf.Mult('Bow2', 0,5)
pmf.Normalize()
