__author__ = 'auroua'
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

count_data = np.loadtxt("txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data)
plt.show()