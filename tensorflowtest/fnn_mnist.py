import cv2
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
img = mnist.train.images[3, :]
print img.shape
cv2.imshow('test', img.reshape((28, 28)))
cv2.waitKey(0)