__author__ = 'auroua'
from thinkbayes import Pmf
class Monty(Pmf):
    def __init__(self, hypos):
        # must
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Updates(self, data):
        for hypo in self.Values():
            like = self.Likeihood(hypo, data)
            self.Mult(hypo, like)
        self.Normalize()

    def Likeihood(self, hypo, data):
        if data==hypo:
            return 0
        elif hypo=='A':
            return 0.5
        else:
            return 1

if __name__=='__main__':
    hyposs = ['A', 'B', 'C']
    pmf = Monty(hyposs)
    pmf.Updates('B')
    for hypo, prob in pmf.Items():
        print hypo, prob