#encoding:UTF-8
__author__ = 'auroua'
import media

values = ['a','b','c']

for p in range(len(values)):
    print p,values[p]

for x in enumerate('abc'):
    print x

for u,v in enumerate(['a','b','c']):
    print u,v

first,second,third = ['a','b','c']

def count_fragments(fragment,dna):
    count = 0
    last_match=0
    while last_match!=-1:
        last_match = dna.find(fragment,last_match)
        if last_match!=-1:
            last_match+=1
            count+=1
    return count

print count_fragments('gtg','gttacgtggatg')
print count_fragments('gtt','gttacgtggatg')

def file_process():
    enter_number = 1
    files = open("fileinput_continue_data.dat",'r')
    for line in files:
        line = line.strip()
        if line.startswith('#'):
            continue
        if line=='Earth':
            break
        enter_number = enter_number + 1
    print enter_number

file_process()


def pic_process():
    pic = media.load_image("/home/auroua/workspace/lena.jpg")
    width,height = media.get_width(pic),media.get_height(pic)
    for x in range(0,height,2):
        for y in range(0,width+1):
            p = media.get_pixel(pic,x,y)
            media.set_color(p,media.black)

    media.show(pic)