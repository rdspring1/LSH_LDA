"""
Implementation of the collapsed Gibbs sampler for Latent Dirichlet Allocation
"""

import numpy as np
import scipy as sp
from scipy.special import gammaln

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class LdaSampler():
    def __init__(self, num_topics, id2word, alpha=None, beta=None):
        """
        num_topics: desired number of topics
        alpha: a scalar
        beta: a scalar
        """
        self.num_topics = num_topics
        uniform_prior = 1.0 / self.num_topics
        self.alpha = uniform_prior if not alpha else alpha
        self.beta = uniform_prior if not beta else beta
        self.id2word = id2word
        self.vocab_size = len(id2word.keys())

    def _initialize(self, bow):
        n_docs = len(bow)

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.num_topics))

        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.num_topics, self.vocab_size))

        # number of words per document
        self.nm = np.zeros(n_docs)

        # number of words per topic
        self.nz = np.zeros(self.num_topics)

        # (document, word) -> topic
        self.topics = dict()

        for m in range(n_docs):
            offset = 0
            for i, w in enumerate(bow[m]):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.num_topics)
                self.nmz[m,z] += 1
                self.nm[m] += 1
                self.nzw[z,w] += 1
                self.nz[z] += 1
                self.topics[(m,i)] = z

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size num_topics).
        """
        # P(w|t) = topic assignment per word / number of words per topic
        left = (self.nzw[:,w] + self.beta) / (self.nz + self.beta * self.vocab_size)
        # P(t|d) = number of words for each topic per document / number of words per document
        right = (self.nmz[m,:] + self.alpha) / (self.nm[m] + self.alpha * self.num_topics)

        p_z = left * right
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in range(self.num_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, self.vocab_size)

        for m in range(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.num_topics)

        return lik

    def topk(self, k=5):
        partition = np.sum(self.nzw, axis=1)
        for topic in range(self.num_topics):
            p_w = self.nzw[topic, :] 
            ids = np.argsort(p_w)
            for word in range(k):
                word_idx = ids[self.vocab_size-word-1]
                print(topic, self.id2word[word_idx], p_w[word_idx] / float(partition[topic]))

    def run(self, bow, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_docs = len(bow)
        self._initialize(bow)

        for it in range(maxiter):
            for m in range(n_docs):
                offset = 0
                for i, w in enumerate(bow[m]):
                    # select topic for current word/document
                    z = self.topics[(m,i)]

                    # create conditional state - remove current word from distribution
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    # add current word to distribution
                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1

                    # update word topic
                    self.topics[(m,i)] = z
