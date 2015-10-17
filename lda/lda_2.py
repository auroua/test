__author__ = 'auroua'
from nltk.corpus import inaugural
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from lda_1 import LDA
import seaborn as sns

stops = set(stopwords.words("english"))

vocab = dict()
for fileid in inaugural.fileids():
    for word in inaugural.words(fileid):
        word = word.lower()
        if word not in stops and word.isalpha():
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

"""
Sort the vocab keep only words which occur more than 50 times
Then Create word to id and id to word dictionaries
"""
vocab_sorted = filter(lambda x: x[1] > 50, sorted(vocab.items(), key=lambda x: x[1], reverse=True))
wordids = {v[0]: i for i, v in enumerate(vocab_sorted)}
idwords = {i: v[0] for i, v in enumerate(vocab_sorted)}
vocab_size = len(wordids)
print vocab_size

# Generate corpus document vectors
data = []
for fileid in inaugural.fileids():
    data.append([0]*vocab_size)
    for word in inaugural.words(fileid):
        word = word.lower()
        if word in wordids:
            data[-1][wordids[word]] += 1

print data
len(data)

print data[0][:10]
data = np.array(data)
plt.clf()
plt.matshow(data, fignum=1000, cmap=plt.cm.Reds)
plt.gca().set_aspect('auto')
plt.xlabel("Words")
plt.ylabel("Documents")
# plt.show()

inaugural_lda = LDA(data, topics=10, vocab=vocab_size)
inaugural_lda.fit()
inaugural_lda.plot_words_per_topic()