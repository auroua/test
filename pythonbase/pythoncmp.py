__author__ = 'auroua'

class MyClass(object):
    def __init__(self, name):
        self.name = name

    def __cmp__(self, other):
        return cmp(self.name, other.name)

a = MyClass('leon')
b = MyClass('leon')
print a is b
print a == b   #compare the content
print id(a)
print id(b)
print cmp(a, b)