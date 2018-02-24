# -*- coding: utf-8 -*-
from __future__ import print_function
from scipy.special import digamma
import numpy as np
from gensim import corpora, models, similarities
import sys

class LDA:
    def __init__(self, n_topic, n_iter, alpha=0.1, beta=0.01):
        self.n_topic = n_topic
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        
    def fit(self, curpus):
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
        
        theta = np.random.uniform(size=(n_documents, self.n_topic))
        old_theta = np.copy(theta)
        phi = np.random.uniform(size=(self.n_topic, n_word_types))
            
        for n in range(self.n_iter):
            sum_phi = []
            for k in range(self.n_topic):
                sum_phi.append(sum(phi[k]))
            ndk = theta
            nkv = np.zeros((self.n_topic, n_word_types))
            for d in range(n_documents):
                sum_theta_d = sum(theta[d])
                prob_d = digamma(theta[d]) - digamma(sum_theta_d)
                ndk[d, :] = 0.
                for w in range(len(word_indexes[d])):
                    word_no = word_indexes[d][w]
                    prob_w = digamma(phi[:, word_no]) - digamma(sum_phi)
                    latent_z = np.exp(prob_w + prob_d)
                    latent_z /= np.sum(latent_z)
                    
                    ndk[d, :] += latent_z * word_counts[d][w]
                    nkv[:, word_no] += latent_z * word_counts[d][w]
            theta = ndk + self.alpha
            phi = nkv + self.beta
            print(n, np.max(theta - old_theta))
            old_theta = np.copy(theta)
        
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


