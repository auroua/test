#encoding: UTF-8
__author__ = 'auroua'

import sys
print '{} this is a format test {}'.format('ttt',sys.argv[0])


a=3
b=5
print a<=b

print b/a
print b//a

length = 5
width = 3

print 'area is',length*width

number=23
guess = int(raw_input('please guess the input'))

if guess==number:
    print 'you guess the right answer'
    print 'but you can not get any prize'
elif guess< number:
    print 'the number you guessed is too small'
else:
    print 'the number you guessed is too large'

print 'Done'

running = True

while running:
    guess = int(raw_input("test again use while"))
    if guess == number:
        print 'win'
        running=False;
    elif guess<number:
        print 'too small'
        break
    else:
        print 'too large'
else:
    print 'running finish'
