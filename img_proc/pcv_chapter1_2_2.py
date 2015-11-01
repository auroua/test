__author__ = 'auroua'

import PIL.Image as Image
import matplotlib.pylab as pylab
import img_tools as tools

if __name__ == '__main__':
    files = tools.get_imlist('/home/auroua/workspace/PycharmProjects/data/pcv_img/avg/')
    img = Image.open(files[0]).convert('L')
    im = pylab.array(img)
    pylab.figure()
    pylab.gray()
    # pylab.axis('equal')
    # pylab.axis('off')
    # pylab.contour(im, origin='image')
    # pylab.imshow(im)
    # pylab.hist(im.flatten(),128)

    pylab.imshow(im)
    print 'please click 3 points'
    x = pylab.ginput(3)
    print 'you clicked : ',x
    pylab.show()