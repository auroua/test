#encoding:UTF-8
__author__ = 'auroua'

def total(values,start=0,end=None):
    if end==None:
        end = len(values)

    sum = 0
    for i in range(0,end):
        sum+=values[i]

    print sum

valuess = [1,2,3,4,5]
total(valuess)

def our_maxx(*values):
    if not values:
        return None
    max = values[0]
    for text in values:
        if max<text:
            max = text
    print max

our_maxx(3,2,4,5,6,7,1,2)
our_maxx()


try:
    x = 1.0/0.3
    print 'reciprocal of 0.3 is',x
    x = 1.0/0.0
    print 'reciprocal of 0.0 is',x
except:
    print 'error:no reciprocal'
else:
    print 'all finished'

values = [-1,0,1]
for i in range(4):
    try:
        r = 1.0/values[i]
        print 'reciprocal of',values[i],'at',i,'is',r
    except IndexError,e:
        print 'index',i,'out of range'
        print 'error is ',e
    except ArithmeticError,e:
        print 'unable to calculate reciprocal of',values[i]
        print 'error is',e

print [x*x for x in range(1,11) if x%3==0]

print [m+n for m in 'ABC' for n in 'EFG']

L = ['Hello', 'World', 18, 'Apple', None]

print [s.lower() for s in L if isinstance(s,str)]

print not []