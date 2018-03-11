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
        
    def fit_transform2(self, curpus):
        word_indexes = []
        word_counts = []
        for row_curpus in curpus:
            row_indexes = []
            row_counts = []
            for w_i, w_c in row_curpus:
                row_indexes.append(w_i)
                row_counts.append(w_c)
            word_indexes.append(row_indexes)
            word_counts.append(row_counts)
        
        n_documents = len(word_indexes)    
        
        max_index = 0
        for d in range(n_documents):
            document_max = np.max(word_indexes[d])
            if max_index < document_max:
                max_index = document_max
                
        n_word_types = max_index + 1
        
        theta = np.random.rand(n_documents, self.n_topic) + self.alpha
        phi = np.random.rand(self.n_topic, n_word_types) + self.beta
        t1 = time.time()
        old_loglikely = False
        old_theta = theta.copy()
        for n in range(self.n_iter):
            dig_alpha = digamma(theta) - digamma(theta.sum(axis = 1, keepdims = True))
            dig_beta = digamma(phi) - digamma(phi.sum(axis = 1, keepdims = True))
            alpha_new = np.ones((n_documents, self.n_topic)) * self.alpha
            beta_new = np.ones((self.n_topic, n_word_types)) * self.beta
            for (d, row_corpus) in enumerate(curpus):
                q = np.zeros((n_word_types, self.n_topic))
                v = np.array(map(lambda x: x[0], row_corpus))
                count = np.array(map(lambda x: x[1], row_corpus))
                q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
                q[v, :] /= q[v, :].sum()
                alpha_new[d, :] += count.dot(q[v])
                beta_new[:, v] += count * q[v].T
                    
                """
                word_no = word_indexes[d][w]
                latent_z = np.exp(prob_w[:, word_no] + prob_d)
                latent_z /= np.sum(latent_z)
                
                ndk[d, :] += latent_z * word_counts[d][w]
                nkv[:, word_no] += latent_z * word_counts[d][w]
                """
                    
            theta = alpha_new.copy()
            phi = beta_new.copy()
            if (n + 1) % 10 == 0:
                tim = time.time() - t1
                t1 = time.time()
                loglikely = self.lhood(theta, phi, curpus)
                if old_loglikely != False:
                    convergence = (old_loglikely - loglikely) /  old_loglikely
                else:
                    convergence = float('inf')
                print("[%d] log likelyhood:%.3f(%.5f) %.1fsec"%(n + 1, loglikely , convergence, tim))
                
                #if old_loglikely != False and convergence < 1e-5:
                #   break
                print("convergence", np.max(np.abs(old_theta - theta)))
                old_theta = theta.copy()
                old_loglikely = loglikely
        
        for k in range(self.n_topic):
            phi[k] = phi[k] / np.sum(phi[k])

        for d in range(n_documents):
            theta[d] = theta[d] / np.sum(theta[d])
            
        return phi, theta
    
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


