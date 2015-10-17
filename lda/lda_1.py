#encoding : UTF-8
__author__ = 'auroua'

import numpy as np
import pymc as pm
#K, V, D = 2, 4, 3 # number of topics, words, documents
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

K, V, D = 5, 10, 20 # number of topics, words, documents

data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

data_temp = np.random.randint(0,10,size=(D,V))

class LDA(object):
    def __init__(self,data,topics=K,vocab=V):
        """
        Takes the data variable and outputs a model
        """
        self.data = data
        self.topics = topics
        self.vocab = vocab+1
        self.docs = len(self.data)
        self.alpha = np.ones(self.topics)
        self.beta = np.ones(self.vocab)

        self.theta = pm.Container([pm.CompletedDirichlet("theta_%s" % i,pm.Dirichlet("ptheta_%s" % i, theta=self.alpha)) for i in range(self.docs)])
        self.phi = pm.Container([pm.CompletedDirichlet("phi_%s" % i,pm.Dirichlet("pphi_%s" % i, theta=self.beta)) for i in range(self.topics)])
        self.Wd = [len(doc) for doc in self.data]
        self.Z = pm.Container([pm.Categorical("z_%s" % d,p=self.theta[d],size=self.Wd[d],value=np.random.randint(self.topics,size=self.Wd[d])) for d in range(self.docs)])
        self.W = pm.Container([pm.Categorical("w_%s,%s" % (d,i),p=pm.Lambda("phi_z_%s_%s" % (d,i),lambda z=self.Z[d][i], phi=self.phi: phi[z]),value=self.data[d][i],
                                              observed=True) for d in range(self.docs) for i in range(self.Wd[d])])
        self.model = pm.Model([self.theta, self.phi, self.Z, self.W])
        self.mcmc = pm.MCMC(self.model)

    def fit(self, iterations=1000, burn_in=10):
        # Fit the model by sampling from the data iterations times with burn in of burn_in.
        self.mcmc.sample(iterations, burn=burn_in)

    def show_topics(self):
        # Show distribution of topics over words
        return self.phi.value

    def show_words(self):
        # Show distribution of words in documents over topics
        return self.W.value

    def KLDiv(self, p,q):
        return np.sum(p*np.log10(p/q))

    def cosine_sim(self, x,y):
        return np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y))

    def sorted_docs_sim(self):
        kldivs_docs = [(i, j, self.KLDiv(self.theta[i].value,self.theta[j].value),
                        self.cosine_sim(self.data[i], self.data[j]))
                       for i in range(len(self.theta)) for j in range(len(self.theta))
                       if i != j]
        return sorted(kldivs_docs, key=lambda x: x[3], reverse=True)

    # def show_topic_words(self, n=10, idwords):
    #     for i, t in enumerate(self.phi.value):
    #         print "Topic %i : " % i, ", ".join(idwords[w_] for w_ in np.argsort(t[0])[-10:] if w_ < (self.vocab-1-1))

    def plot_data(self):
        plt.clf()
        plt.matshow(data, fignum=1000, cmap=plt.cm.Reds)
        plt.gca().set_aspect('auto')
        plt.xlabel("Words")
        plt.ylabel("Documents")

    def plot_words_per_topic(self, ax=None):
        if ax is None:
            plt.clf()
            fig, ax = plt.subplots(1,1)
        words = self.Z.value
        print words
        topic_dist = dict()
        for k_i in words:
            for k in k_i:
                if k not in topic_dist:
                    topic_dist[k] = 0
                topic_dist[k] += 1
        ax.bar(topic_dist.keys(), topic_dist.values())
        ax.set_xlabel("Topics")
        ax.set_ylabel("Counts")
        ax.set_title("Document words per topics")
        plt.show()

    def plot_word_dist(self, ax=None):
        topics = self.phi.value
        print topics
        if ax is None:
            plt.clf()
            fig, ax = plt.subplots((len(topics)+1)/2, 2, figsize=(10,10))
        for i, t in enumerate(topics):
            ax[i/2][i%2].bar(range(len(t[0])), t[0])
            ax[i/2][i%2].set_title("Topic %s" % i)
        plt.suptitle("Vocab word proportions per topic")
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show()


if __name__ == '__main__':
    lda = LDA(data_temp)
    lda.fit()
    # lda.plot_words_per_topic()
    lda.plot_word_dist()