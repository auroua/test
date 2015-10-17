#encoding:UTF-8
__author__ = 'auroua'

import sys

ten = set(range(10))
lows = set([0,1,2,3,4,5])

lows.add(9)
#print lows
lows.remove(0)

birds = set()
files = open("birdwatching.txt",'r')
for lines in files:
    birds.add(lines.strip())
files.close()

#print birds

birds2 = {'canada goose':3,'northern fulmar':1}
print birds2
print birds2['canada goose']

birds2['northern fulmar'] = 100
del birds2['canada goose']

for x in birds2:
    print x,birds2[x]

print birds2.keys()
print birds2.values()
print birds2.items()

for x,v in birds2.items():
    print x,v

def get_total_birds(filename):
    total_birds = {}
    files2 = open(filename,'r')
    for lines in files2:
        name = lines.strip()
        total_birds[name] = total_birds.get(name,0)+1
    print total_birds
    return total_birds

def inver_total_birds(dict_birds):
    invert_birds = {}
    for name,value in dict_birds.items():
        if value in invert_birds:
            invert_birds[value].append(name)
        else:
            invert_birds[value] = [name]
    print invert_birds
    return invert_birds

#inver_total_birds(get_total_birds("birdwatching.txt"))

sys.argv[1]

def compand_dict(r):
    compand_list = []
    while True:
        liness = r.readline()
        if liness and not liness.startswith("END"):
            name,code,short,x,y,z = liness.split()
            compand_list.append([name,code,short,x,y,z])
        else:
            break

    if liness.startswith("END"):
        liness = r.readline()
    return compand_list,liness

def read_all_molecules(r):
    compands = {}
    lines = r.readline()
    oldlines = lines
    while lines:
        molecule,lines = compand_dict(r)
        fields = oldlines.split()
        oldlines = lines
        compands[fields[1]] = molecule
    print compands

filesss = open("multimol.pdb",'r')
read_all_molecules(filesss)

