#encoding:UTF-8
import os
from PIL import Image
import numpy as np
from matplotlib import pylab


def get_imlist(path):
    '''return the absolute path of the image files in the path folder'''
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    pil_im = Image.fromarray(np.uint8(im))
    return pylab.array(pil_im.resize(sz))


def histeq(im, nbr_bins=256):
    """对一副图像进行直方图均衡化"""
    imhist, bins = pylab.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255.0*cdf/cdf[-1]
    im2 = pylab.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    """计算图像列表的平均图像"""
    averageim = np.array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print imname + '...skipped'
    averageim /= len(imlist)

    return np.array(averageim, 'uint8')


if __name__ == '__main__':
    empire_im = pylab.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/empire.jpg').convert('L'))
    pylab.gray()
    pylab.imshow(empire_im)
    pylab.figure()
    im22, cdfo = histeq(empire_im)
    im2_img = Image.fromarray(np.uint8(im22))
    pylab.imshow(im2_img)
    pylab.figure()
    array_img2 = pylab.array(np.uint8(im2_img))
    # pylab.hist(array_img2.flatten(), 256)
    pylab.hist(cdfo, 256)
    pylab.show()