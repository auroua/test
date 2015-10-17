__author__ = 'auroua'

fs = [(lambda x:x+y) for y in range(10)]

print fs[3](4)

print [f(4) for f in fs]

fs = []

for i in range(10):
    def fun(x):
        return x+i;
    fs.append(fun)

print [f(4) for f in fs]

fs2 = []
for i in range(10):
    def g(n,i=i):
        return n+i;
    fs2.append(g)

print [ff(3) for ff in fs2]

fss = [lambda n,i=i:n+i for i in range(10)]

print [f(3) for f in fss]

f1,f2,f3 = [lambda n,i=i:n+i for i in range(1,4)]

print f1(1),f2(1),f3(1)

def count():
    fs = []
    for i in range(1,4):
        def g():
            return i*i
        fs.append(g)

    return fs

ffss = count()
print [f() for f in ffss]


def test_a(x):
    return x+1

testb = 3

print test_a(testb),testb