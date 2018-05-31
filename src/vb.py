# -*- coding: utf-8 -*-
from __future__ import print_function
from scipy.special import digamma
import numpy as np
from gensim import corpora, models, similarities
import sys
import time

class LDA:
    def __init__(self, n_topic, n_iter, alpha=0.5, beta=0.5):
        self.n_topic = n_topic
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta

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
        
        doc_v = []
        doc_count = []
        for (d, row_corpus) in enumerate(curpus):
            doc_v.append(np.array(map(lambda x: x[0], row_corpus)))
            doc_count.append(np.array(map(lambda x: x[1], row_corpus)))
        
        alpha = self.alpha + np.random.rand(D, K)
        beta = self.beta + np.random.rand(K, V)
        # estimate parameters
        old_alpha = alpha.copy()
        for t in range(self.n_iter):
            dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
            dig_beta = digamma(beta) - digamma(beta.sum(axis = 1, keepdims = True))

            alpha_new = np.ones((D, K)) * self.alpha
            beta_new = np.ones((K, V)) * self.beta
            for (d, row_corpus) in enumerate(curpus):
                q = np.zeros((V, K))
                v = doc_v[d]
                count = doc_count[d]
                q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v])).T
                q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)

                # alpha, beta
                alpha_new[d, :] += count.dot(q[v])
                beta_new[:, v] += count * q[v].T
            alpha = alpha_new.copy()
            beta = beta_new.copy()
            print(t, self.lhood(alpha, beta, curpus))
            print("convergence", np.max(np.abs(old_alpha - alpha)))
            old_alpha = alpha.copy()
            
        for k in range(self.n_topic):
            beta[k] = beta[k] / np.sum(beta[k])

        for d in range(D):
            alpha[d] = alpha[d] / np.sum(alpha[d])
            
        return beta, alpha
        
    
if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    from pprint import pprint
    
    newsgroups_train = fetch_20newsgroups(subset='train')
    
    skip_headers = ["Nntp-Posting-Host:", "From:", "Organization:", "Lines:"]
    delete_header_names = ["Subject: ", "Summary: ", "Keywords: "]
    documents = []
    for message in newsgroups_train.data[:1000]:
        content = ""
        for line in message.split("\n"):
            is_skip = False
            for header in skip_headers:
                if line[:len(header)].lower() == header.lower():
                    is_skip = True
                    break
            if is_skip:
                continue
            content += line + "\n"
        content = content.strip()
        for header in delete_header_names:
            content = content.replace(header, "")
        documents.append(content)
            
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
                for document in documents]
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once]
                for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    n_topics = 20
    lda = LDA(n_topics, 5000)
    phi, theta = lda.fit(corpus)
    for k in range(n_topics):
        print("topic:", k)
        indexes = np.argsort(phi[k])
        for word in indexes[::-1][:30]:
            print(dictionary[word])
        print("")


