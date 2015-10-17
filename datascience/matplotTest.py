#encoding:UTF-8
__author__ = 'auroua'
import matplotlib.pylab as plt
import numpy as np
fig , axes = plt.subplots(2,3)
for i in range(2):
    for j in range(3):
        if i==1 and j==2:
            axes[1,2].plot(np.random.randn(30).cumsum(),'ko--',label='Default')
        else:
            axes[i,j].hist(np.random.randn(500),bins=50,color='k',alpha=0.5)


plt.legend('best')

plt.show()