__author__ = 'auroua'

def _private():
    print 'hello private'

def __private1():
    print 'hello private1'

def test():
    _private()
    __private1()


