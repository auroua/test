#encoding: UTF-8
__author__ = 'auroua'

class Color(object):
    def __init__(self,blue,green,red):
        '''A new Color will be constructed'''
        self.blue = blue
        self.green = green
        self.red = red
    def __str__(self):
        '''return the string format of this class'''
        return "Color is green {0}, blue {1}, red {2}".format(self.green,self.blue,self.red)
        #return "Color is red=%s,green=%s,blue=%s" %(self.red,self.green,self.blue)

    def lightness(self):
        '''return the normal color'''
        strongness = max(self.blue,self.green,self.red)
        lightness = min(self.blue,self.green,self.red)
        return (0.5*(strongness+lightness))/255

    def __add__(self, other):
        '''return the added result of two color classd'''
        return Color(min(self.blue+other.blue,255),
                     min(self.green+other.green,255),
                     min(self.red+other.red,255))

    def __sub__(self, other):
        return Color(abs(self.blue-other.blue),
                     abs(self.green-other.green),
                     abs(self.red-other.red))

    def __eq__(self, other):
        return self.blue==other.blue and self.green==other.green and self.red==other.red

black = Color(232,33,123)
black2 = Color(232,33,121)

black_n = black - black2
print black_n
print black==black2

print black
print black.lightness()

print dir(black_n)

print help(black_n)

print black_n.__dict__

class Organism(object):
    '''有机物的python类'''
    def __init__(self,name,x,y):
        self.name = name
        self.x = x
        self.y = y

    def __str__(self):
        '''the string format of this object'''
        return 'the name of this object is {},and location of this object is (x,y),({},{})'.format(self.name,self.x,self.y)

    def can_eat(self):
        return False

    def move(self):
        return False

class Arthropod(Organism):
    def __init__(self,name,x,y,legs):
        Organism.__init__(self,name,x,y)
        self.legs = legs

blue_crap = Arthropod("Callinectes",0,0,8)
print blue_crap


class Atom(object):
    def __init__(self,number,sym,x,y,z):
        '''construct a new atom'''
        self.number = number
        self.center = [x,y,z]
        self.sym = sym

    def __str__(self):
        '''show in string format'''
        return 'the No. of the atom is {},sym of this atom is {},the construct of this atom is {}'.format(self.number,self.sym,self.center)

    def translate(self,x,y,z):
        '''the move of the atom'''
        temp = float(self.center[0])
        temp = temp + float(x)
        self.center[0] = temp
        temp = float(self.center[1])
        temp = temp + float(y)
        self.center[1] = temp
        temp = float(self.center[2])
        temp = temp + float(z)
        self.center[2] = temp

class Molecule(object):
    def __init__(self,name):
        self.name = name
        self.atoms=[]

    def add_atom(self,atom):
        '''add an atom to the molecule'''
        self.atoms.append(atom)

    def __str__(self):
        '''to string format'''
        f_str = '\n'
        for atom in self.atoms:
            f_str+=str(atom)
            f_str+='\n'

        return 'the molecule name is {} and atom is {} '.format(self.name,f_str)

    def translates(self,x,y,z):
        for atmo in self.atoms:
            atmo.translate(x,y,z)

def read_molecule(r):
    line = r.readline()

    if not line:
        return None

    index,name = line.split()
    molecule = Molecule(name)

    reading = True

    while reading:
        line = r.readline()
        if not line.startswith('END'):
            name,num,sym,x,y,z = line.split()
            molecule.add_atom(Atom(num,sym,x,y,z))
        else:
            reading = False

    return molecule

files = open('pythonbase/ammonia.pdb','r')

moslue = read_molecule(files)

print moslue

moslue.translates(3,2,1)

print moslue


print (lambda :3)()

print (lambda x:2*x**2)(3)

#lambda的函数形式
print (lambda : read_molecule(files))()

g = (x*x for x in range(0,100))

print  g.next()
print  g.next()

for y in g:
    print y

def fib(maximum):
    n,a,b = 0,0,1

    while n<maximum:
        #print b
        #generator
        yield b
        a,b = b,a+b
        n += 1

o = fib(6)
print o.next()

def test_yield():
    print 'step 1'
    yield 1
    print 'step 2'
    yield 2
    print 'step 3'
    yield 3
oo = test_yield()
print oo.next()
print oo.next()

def get_generator_value(gen,index):
    n = 0
    for indexs in gen:
        if n == index:
            return indexs
        n+=1

print get_generator_value(fib(6),3)

print abs

f = abs
print f(-10)
#返回内存地址
print id(f)