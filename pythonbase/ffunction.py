#encoding:UTF-8
__author__ = 'auroua'

import functools
import testprivate

testprivate.__private1()

print '===================='
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

try:
    import json
except ImportError:
    import simplejson as json
print int('123456')

print int('123456',base=8)

def int2(x,base=2):
    return int(x,base)

print int2('1001001')

int3 = functools.partial(int,base=2)

print int3('000111')


kw = {'base':2}

print int('10101',**kw)

max2 = functools.partial(max,10)
print max2(4,5,6,7)

import sys
print sys.path

import __future__
print dir(__future__)

class Student(object):
    def __init__(self,name,no,age):
        self.name = name
        self.no = no
        self.__age = age

    def __str__(self):
        return "student name is {} and no is {} and age is {}".format(self.name,self.no,self.__age)

zs = Student('zhangsan','12','13')

zs._Student__age = 23

print zs


class Animal(object):
    __slots__ = ('x','y','z')
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    def run(self):
        print "Animal run"

class Dog(Animal):
    def __init__(self):
        pass

    def run(self):
        print "dog run"

class Cat(Animal):
    def __init__(self):
        pass

    def run(self):
        print "cat run"


def run_twice(animal):
    animal.run()
    animal.run()

dd = Dog()
cc = Cat()

dd.run()
cc.run()

print isinstance(dd,Animal)

print type(dd)

print len('ABC')

print 'ABC'.__len__()


aa = Animal(12,13,14)

# if hasattr(aa,'f'):
#     getattr(aa,'f')
# else:
#     setattr(aa,'f',15)
#aa.f=5
