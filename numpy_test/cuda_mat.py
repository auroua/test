__author__ = 'auroua'
import numpy as np
import cudamat as cm

cm.cublas_init()

# create two random matrices and copy them to the GPU
a = cm.CUDAMatrix(np.random.rand(32, 256))
b = cm.CUDAMatrix(np.random.rand(256, 32))

# perform calculations on the GPU
c = cm.dot(a, b)
d = c.sum(axis = 0)

# copy d back to the host (CPU) and print
print(d.asarray())


a1 = np.random.rand(32, 256)
b2 = np.random.rand(256, 32)

# perform calculations on the GPU
c = np.dot(a1, b2)
d = c.sum(axis = 0)

print d