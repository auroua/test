import numpy as np
import matplotlib.pylab as plt

x1 = np.random.uniform(0, 1, 1000)
x2 = np.random.uniform(0, 1, 1000)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.hist(x1, bins=20, color='k', alpha=0.3)
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(x2, bins=20, color='g', alpha=0.3)
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(x1+x2, bins=20, color='b', alpha=0.3)
plt.show()