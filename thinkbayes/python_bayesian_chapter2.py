__author__ = 'auroua'
from thinkbayes import Pmf
class Cookie(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in hypos:
            like = self.Likeihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    mixs = {'bow1' : dict(vanilla=0.75, chocolate=0.25),
            'bow2' : dict(vanilla=0.5, chocolate=0.5)
           }

    def Likeihood(self, data, hypo):
        return self.mixs[hypo][data]

if __name__ == '__main__':
    hypos = ['bow1', 'bow2']
    pmf = Cookie(hypos)
    print pmf.Prob('bow1')
    pmf.Update('vanilla')
    print pmf.Prob('bow1')