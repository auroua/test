#encoding:UTF-8
__author__ = 'auroua'

def skip_head(files):
    line = files.readline()
    line = files.readline()
    while line.startswith('#'):
        line = files.readline().strip()
    return line
#
#
# input_file = open("hopedale.txt","r")
# # first_line = skip_head(input_file)
# # print first_line
# # for line in input_file:
# #     line = line.strip()
# #     print line
#
#
# def find_smallest(files):
#     small = int(skip_head(files))
#
#     for newlines in files:
#         if int(newlines)<small:
#             small = int(newlines)
#     print small
#
# print '####################'
# find_smallest(input_file)
#
#
# input_file.close()

input_file2 = open("lynx.txt",'r')
line = skip_head(input_file2)

def find_largest(lines):
    largest = -1

    for value in lines.split():
        if int(value[0:-1])>largest:
            largest = value[0:-1]

    return largest

largests = find_largest(line)

for lines in input_file2:
    if largests<find_largest(lines):
        largests = find_largest(lines)

print largests

input_file3 = open("housing.dat")

starts = []
contacts = []
rates = []
for lines in input_file3:
     start,contact,rate=lines.split()
     starts.append(float(start))
     contacts.append(float(contact))
     rates.append(rate)

print starts
print contacts
print rates

print sum(starts[12:24])-sum(starts[0:12])

def read_weather_data(r):
    fields = ((4,int),(2,int),(2,int),
              (2,int),(2,int),(2,int),
              (2,int),(2,int),(2,int),
              (6,int),(6,int),(6,int))

    results = []
    for line in r:
        start = 0
        record = []
        for (width,target_type) in fields:
            text = line(start,start+width)
            record.append(target_type(text))
            start+=width
        results.append(record)

    return results