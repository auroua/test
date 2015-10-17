# encoding: utf-8
__author__ = 'auroua'

whales = [5,4,7,3,3,2,6,5,1]

print whales[-3:]

krypton = ['Krypton','Kr',-157.2,-153.4]
print krypton[1]
print krypton[2]

whales[1] = 3

def printall(lists):
    print '####################'
    for p in lists:
        print p
    print '####################'

printall(whales)

print len(whales),3
print sum(whales)
print max(whales)
print min(whales)

newkrypton = krypton + ['test']

printall(newkrypton)

metal = 'Fe Ni'.split()
print metal
print metal*3

outer = ['Li','Na','K']
inner = ['F','C1','Br']

for metal in outer:
    for halo in inner:
        print metal+halo

def print_tables():
    '''print out the multiplication table for numbers 1 through 5'''
    numbers = [1,2,3,4,5]

    for i in numbers:
        print '\t'+str(i),
    print

    for i in numbers:
        print str(i),
        for j in numbers:
            print '\t'+str(i*j),
        print


print_tables()

krypton.append('ttt')
printall(krypton)

def generate_emb_lists():
    temp = []
    final = []
    #注意别名的问题  temp始终是一个变量
    # for i in range(0,3):
    #     temp.append(i)
    # for i1 in range(0,3):
    #     final.append(temp)
    for i in range(0,10):
        final.append(range(0,10))
    for i2 in range(0,len(final)):
        for j in range(0,len(final[i2])):
            final[i2][j] **= i2
    return final

def printlists(embadedlists):
    for i in range(len(embadedlists)):
        print str(i),
        for j in range(len(embadedlists[i])):
            print '\t'+str(embadedlists[i][j]),
        print


printlists(generate_emb_lists())

rawString = r'test /t'

print rawString