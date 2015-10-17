#encoding:UTF-8
__author__ = 'auroua'

def f(x):
    return x**2

print map(f,[1,2,3,4,5,6])

print map(str,[1,2,3,4,5])

def add(x,y):
    return x+y

print reduce(add,[1,2,3,4,5,6])

def str2int(strs):
    def fn(x,y):
        return x*10+y
    def char2num(s):
        return {'0':0,'1':1,'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
    #return reduce(fn,map(char2num,strs))
    return reduce(lambda x,y:x*10+y,map(char2num,strs))

print str2int('232323')

def strprocess(names):
    first = names[0].upper()
    second = names[1:].lower()
    return str(first)+str(second)

print map(strprocess,['adam', 'LISA', 'barT'])

def multiple(x,y):
    return x*y

print reduce(multiple,[1,2,3,4,5,6])

def is_odd(n):
    if n%2==0:
        return True
    else:
        return False

print filter(is_odd,[1,2,3,4,5,6,7,8,9])

def not_empty(s):
    return s and s.strip()

print filter(not_empty,['a','','c'])

def is_sushu(n):
    if n==2:
        return False
    elif n==3:
        return False
    elif n==5:
        return False
    elif n==7:
        return False
    else:
        return (n%2==0) or (n%3==0) or (n%5==0) or (n%7==0)

print filter(is_sushu,range(1,100))


print sorted([32,22,31,24,26,65])

def reserved_sorted(x,y):
    if x>y:
        return -1
    elif x==y:
        return 0
    else:
        return 1

print sorted([32,22,31,24,26,65],reserved_sorted)

def lazy_sum(*arg):
    def sum():
        ax=0
        for x in arg:
            ax+=x
        return ax
    return sum

f = lazy_sum(1,2,34,5,6,7)

print f()

print '#############'

print range(1,4)

def count():
    fs = []
    for i in range(1,4):
        def sum():
            return i*i
        fs.append(sum)
    return fs

f1,f2,f3=count()
print f1()
print f2()
print f3()

def count():
    fs = []
    for i in range(1,4):
        def g(j):
            def f():
                return j*j;
            return f
        fs.append(g(i))
    return fs

f1,f2,f3 = count()
print f1()
print f2()
print f3()

f = [(lambda n:n+i) for i in range(10)]

print f[3](4)
print f[4](4)
print f[5](4)

f = [(lambda n,i=i:n+i) for i in range(10)]

print f[3](4)
print f[4](4)
print f[5](4)