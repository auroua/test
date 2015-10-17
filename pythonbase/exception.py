#encoding:UTF-8
__author__ = 'auroua'

import os
import logging
import json

def foo(s):
    assert int(s)!=0,'divided by zero'
    return 10/int(s)

def bar(s):
    return foo(s)*2

def main():
    try:
        bar('0')
    except StandardError,e:
        logging.exception(e)


print 'END'
print os.name

class Student(object):
    def __init__(self,name,age,gender):
        self.name = name
        self.age = age
        self.gender = gender


def Student2dict(std):
    return {
        'name':std.name,
        'age':std.age,
        'gender':std.gender
    }

def dict2student(d):
    return Student(d['name'],d['age'],d['gender'])

zhangsan = Student('zhangsan',32,'male')

print  json.dumps(zhangsan,default=Student2dict)
print  json.dumps(zhangsan,default=lambda obj:obj.__dict__)

str = '{"gender": "male", "age": 32, "name": "lisi"}'
print json.loads(str,object_hook=dict2student)


