# -*- coding: utf-8 -*-
from __future__ import print_function
from scipy.special import digamma
import numpy as np
from gensim import corpora, models, similarities
import sys
from numpy.random.mtrand import beta

class LDA:
    def __init__(self, n_topic, n_iter, inner_iter=5, alpha=0.1, beta=0.01, step_size=0.1):
        self.n_topic = n_topic
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.inner_iter = inner_iter
        self.step_size = step_size
        
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
            d = np.random.randint(0, n_documents)
            n_word_in_doc = len(word_indexes[d])
            sum_phi = []
            for k in range(self.n_topic):
                sum_phi.append(sum(phi[k]))
            
            theta[d, :] = float(n_word_in_doc) / self.n_topic + self.alpha
            for n2 in range(self.inner_iter):
                nkv = np.zeros((self.n_topic, n_word_types))
                ndk = np.zeros(self.n_topic)
                sum_theta_d = sum(theta[d])
                prob_d = digamma(theta[d]) - digamma(sum_theta_d)
                for w in range(n_word_in_doc):
                    word_no = word_indexes[d][w]
                    prob_w = digamma(phi[:, word_no]) - digamma(sum_phi)
                    latent_z = np.exp(prob_w + prob_d)
                    latent_z /= np.sum(latent_z)
                    
                    ndk += latent_z * word_counts[d][w]
                    nkv[:, word_no] += latent_z * word_counts[d][w]
                theta[d] = ndk + self.alpha
                
            difference = (n_documents) * nkv + self.beta - phi
            phi += self.step_size *  difference
            
            print(n, np.max(theta - old_theta))
            old_theta = np.copy(theta)
        
        for k in range(self.n_topic):
            phi[k] = phi[k] / np.sum(phi[k])

        for d in range(n_documents):
            theta[d] = theta[d] / np.sum(theta[d])
            
        return phi, theta
    
if __name__ == "__main__":
    import MySQLdb
    np.random.seed(12345)
    connection = MySQLdb.connect(db="similar_words",user="root",passwd="password")
    connection.set_character_set('utf8')
    cursor = connection.cursor()
    cursor.execute("SELECT search_content FROM documents ORDER BY id LIMIT 10000")
    documents = []
    for search_content, in cursor.fetchall():
        documents.append(search_content)
        
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
    lda = LDA(n_topics, 1500)
    phi, theta = lda.fit(corpus)
    for k in range(n_topics):
        print("topic:", k)
        indexes = np.argsort(phi[k])
        for word in indexes[::-1][:30]:
            print(dictionary[word])
        print("")


