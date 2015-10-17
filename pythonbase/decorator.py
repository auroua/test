#encoding:UTF-8
__author__ = 'auroua'

import functools

def log(func):
    def wrapper(*args,**kw):
        print func.__name__
        return func(*args,**kw)
    return wrapper

@log
def func():
    print "test"

func()

print '======================='

def log2(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args,**kw):
            print text,func.__name__
            return func(*args,**kw)
        return wrapper
    return decorator

@log2("ttttt")
def test():
    print 'very useful'

test()

def test2():
    print 'very useful'

ff = log2('tttttt')(test2)

ff()

print ff.__name__