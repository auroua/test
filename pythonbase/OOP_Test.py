#encoding:UTF-8
__author__ = 'auroua'

class Student:
    def __init__(self,name,age,id):
        self.name = name
        self.age = age
        self.id = id

    @property
    def id(self):
        return id

    # @id.setter
    # def set_id(self,id):
    #     pass

    def get_Name(self):
        return self.name

    def set_Name(self,name):
        if len(name)==0:
            pass
        else:
            self.name = name

    def get_age(self):
        return self.age

    def set_age(self,age):
        if not isinstance(age,int):
            raise ValueError("you should input an integer")
        elif age<0 or age>200:
            raise ValueError("you inputed number is not correct")
        self.age = age

zhangsan = Student('zhangsan',32,10)

zhangsan.name = 'tttt'

zhangsan.set_Name('lisi')

# zhangsan.set_age(1000)


zhangsan.id = 1000

print zhangsan.id


class Fif(object):
    def __init__(self):
        self.a = 0
        self.b = 1

    def __iter__(self):
        return self

    def next(self):
        self.a,self.b = self.b,self.a+self.b
        if self.a>1000:
            raise StopIteration
        return self.a

    def __getitem__(self, item):
        if isinstance(item,int):
            a,b = 1,1
            for x in range(item):
                a,b = b,a+b
            return a
        elif isinstance(item,slice):
            items = []
            start = item.start
            end = item.stop
            a,b = 1,1
            for x in range(end):
                a,b = b,a+b
                if x>=start:
                    items.append(a)
            return items

tt = Fif()
# for n in tt:
#     print n

print tt[3]
print tt[5]

print tt[1:7]

class Student(object):
    def __init__(self,age):
        self.age = age

    def __getattr__(self, item):
        if item=='name':
            return 'zhangsan'

    def __call__(self, *args, **kwargs):
        print 'age is ',self.age
t = Student(32)

t.age
print t.name

t()

print callable(Student)
print callable(Fif)

print '=============='

class Hello(object):
    def hello(self,world='world'):
        print 'hello {}'.format(world)

h = Hello()
h.hello()

print type(Hello)

print type(h)

# class listMetaClass(type):
#     def __new__(cls, *args, **kwargs):
#         kwargs['add'] = lambda self,value:self.appand(value)
#         return type.__new__(cls, *args, **kwargs)

# metaclass是创建类，所以必须从`type`类型派生：
class ListMetaclass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)

class MyList(list):
    __metaclass__ = ListMetaclass # 指示使用ListMetaclass来定制类

mylist = MyList()
mylist.add(1)

print mylist

