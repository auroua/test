#encoding:UTF-8
__author__ = 'auroua'

import os

def get_imlist(path):
    '''return the absolute path of the image files in the path folder'''
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

if __name__ == '__main__':
    print get_imlist('/home/auroua/workspace/PycharmProjects/data/N20040103G/')