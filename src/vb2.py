from __future__ import print_function
import numpy as np
from scipy.special import digamma
import sys

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

def lhood(theta, phi, W, n_documents, n_topic):
    phi_hat = np.zeros(phi.shape)
    theta_hat = np.zeros(theta.shape)
    for k in range(n_topic):
        phi_hat[k] = phi[k] / np.sum(phi[k])
    for d in range(n_documents):
        theta_hat[d] = theta[d] / np.sum(theta[d])
        
    ret = 0.
    for d in range(n_documents):
        v, count = np.unique(W[d], return_counts = True)
        for w, c in zip(v, count):
            word_no = w
            prob = np.dot(theta_hat[d], phi_hat[:, word_no])
            ret += np.log(prob) * c
    return ret

class LDA:
    def __init__(self, n_topic, n_iter, alpha=0.2, beta=0.1):
        self.n_topic = n_topic
        self.n_iter = n_iter
        self.alpha0 = alpha
        self.beta0 = beta
    
    def lhood(self, theta, phi, curpus):
        phi_hat = np.zeros(phi.shape)
        theta_hat = np.zeros(theta.shape)
        n_documents = len(curpus)
        for k in range(self.n_topic):
            phi_hat[k] = phi[k] / np.sum(phi[k])
        for d in range(n_documents):
            theta_hat[d] = theta[d] / np.sum(theta[d])
            
        ret = 0.
        for d, row_corpus in enumerate(curpus):
            for w_i, w_c in row_corpus:
                prob = np.dot(theta_hat[d], phi_hat[:, w_i])
                ret += np.log(prob) * w_c
        return ret
        
    def fit_transform(self, curpus):
        D = len(curpus)
        K = self.n_topic
        
        max_index = 0
        for row_corpus in curpus:
            document_max = 0
            for w_i, w_c in row_corpus:
                if document_max < w_i:
                    document_max = w_i
                    
            if max_index < document_max:
                max_index = document_max
        V = max_index + 1
        print(D, K, V)
        
        alpha = self.alpha0 + np.random.rand(D, K)
        beta = self.beta0 + np.random.rand(K, V)
        theta = normalized_random_array(D, K)
        phi = normalized_random_array(K, V)
        # estimate parameters
        for t in range(self.n_iter):
            dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
            dig_beta = digamma(beta) - digamma(beta.sum(axis = 1, keepdims = True))

            alpha_new = np.ones((D, K)) * self.alpha0
            beta_new = np.ones((K, V)) * self.beta0
            for (d, row_corpus) in enumerate(curpus):
                q = np.zeros((V, K))
                v = np.array(map(lambda x: x[0], row_corpus))
                count = np.array(map(lambda x: x[1], row_corpus))
                q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
                q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)

                # alpha, beta
                alpha_new[d, :] += count.dot(q[v])
                beta_new[:, v] += count * q[v].T
            alpha = alpha_new.copy()
            beta = beta_new.copy()
            print(t, self.lhood(alpha, beta, curpus))
            
if __name__ == "__main__":
    np.random.seed(123456)
    # initialize parameters
    D, K, V = 1000, 2, 6
    alpha0, beta0 = 1.0, 1.0
    alpha = alpha0 + np.random.rand(D, K)
    beta = beta0 + np.random.rand(K, V)
    theta = normalized_random_array(D, K)
    phi = normalized_random_array(K, V)

    # for generate documents
    _theta = np.array([theta[:, :k+1].sum(axis = 1) for k in range(K)]).T
    _phi = np.array([phi[:, :v+1].sum(axis = 1) for v in range(V)]).T

    # generate documents
    W, Z = [], []
    N = np.random.randint(100, 300, D)
    for (d, N_d) in enumerate(N):
        Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis = 1))
        W.append((np.random.rand(N_d, 1) < _phi[Z[-1], :]).argmax(axis = 1))
 
    corpus = []
    for (d, N_d) in enumerate(N):
        v, count = np.unique(W[d], return_counts = True)
        corpus.append(zip(v, count))
    print(D, K, V)
    lda = LDA(2, 100)
    lda.fit_transform(corpus)
    sys.exit(0)
    # estimate parameters
    T = 300
    for t in range(T):
        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
        dig_beta = digamma(beta) - digamma(beta.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new = np.ones((K, V)) * beta0
        for (d, _) in enumerate(N):
            row_corpus = corpus[d]
            # q
            q = np.zeros((V, K))
            v = np.array(map(lambda x: x[0], row_corpus))
            count = np.array(map(lambda x: x[1], row_corpus))
            
            #v, count = np.unique(W[d], return_counts = True)
            q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
            q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)

            # alpha, beta
            alpha_new[d, :] += count.dot(q[v])
            beta_new[:, v] += count * q[v].T
            
        alpha = alpha_new.copy()
        beta = beta_new.copy()
        print(lhood(alpha, beta, W, D , K))
    theta_est = np.array([np.random.dirichlet(a) for a in alpha])
    phi_est = np.array([np.random.dirichlet(b) for b in beta])
