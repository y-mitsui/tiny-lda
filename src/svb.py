# -*- coding: utf-8 -*-
from scipy.special import digamma
import numpy as np
from gensim import corpora, models, similarities
import sys

class LDA:

    def __init__(self, n_topic, n_iter, alpha=0.1, beta=0.01, n_batch=10):
        self.n_topic = n_topic
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.n_batch = n_batch
        
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
        latent_z = []
        for i in range(n_documents):
            latent_z.append(np.zeros((len(word_indexes[i]), self.n_topic)))
            
        for n in range(self.n_iter):
            random_idx = np.random.randint(0, n_documents, size=self.n_batch)
            sum_phi = []
            for k in range(self.n_topic):
                sum_phi.append(sum(phi[k]))
            
            for d in random_idx:
                sum_theta_d = sum(theta[d])
                diga_theta = digamma(theta[d]) - sum_theta_d
                for w in range(len(word_indexes[d])):
                    word_no = word_indexes[d][w]
                    k_sum = 0.
                    for k in range(self.n_topic):
                        prob_w = digamma(phi[k][word_no]) - digamma(sum_phi[k])
                        prob_d = diga_theta[k]
                        latent_z[d][w][k] = np.exp(prob_w + prob_d)
                        k_sum += latent_z[d][w][k]
                    latent_z[d][w] /= k_sum
                    
                for k in range(self.n_topic):
                    theta[d, k] = (latent_z[d][:, k] * word_counts[d]).sum() + self.alpha
            
            for k in range(self.n_topic):
                for v in range(n_word_types):
                    tmp = 0.
                    for d in range(n_documents):
                        index = np.where(np.array(word_indexes[d]) == v)[0]
                        if index.shape[0] == 0:
                            continue
                        
                        target_word_counts = np.array(word_counts[d])[index[0]]
                        tmp += latent_z[d][index, k] * target_word_counts
                    phi[k][v] = tmp + self.beta
            print np.max(theta - old_theta)
            old_theta = np.copy(theta)
            #print phi
            #print theta
            #print latent_z
            #exit(1)
        
        for k in range(self.n_topic):
            phi[k] = phi[k] / np.sum(phi[k])

        for d in range(n_documents):
            theta[d] = theta[d] / np.sum(theta[d])
            
        return phi, theta
            
np.random.seed(12345)
documents = [
             "LSI LDA 手軽 試せる gensim 使った 自然 言語 処理 入門",
             "単語 ベクトル化 する word2vec gensim LDA 使い 指定 二単語間 関連",
             "word2vec 仕組み gensim 使う 文書 類似 度 算出 チュートリアル",
             "機械学習 これ 始める 人 押さえる ほしい こと",
             "初心者 向け 機械学習 ディープラーニング 違い シンプル 解説",
             "機械学習 データサイエンティスト 機械学習 ディープラーニング エンジニア なる スキル 要件",
             "セクハラ やじ浴びた 前 都議 民進党 衆院 選 ",
             "執行部 成立 させる なくなる 民進党 内ゲバ 離党 ドミノ 衆院 選",
             "前原 代表 選 民進党 再生 できる"
             ]
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in documents]
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
            for text in texts]
print texts
dictionary = corpora.Dictionary(texts)
print dictionary.token2id
new_doc = "Human computer interaction"
corpus = [np.array(dictionary.doc2bow(text)) for text in texts]
n_topics = 3
lda = LDA(n_topics, 10)
docs_w = [[1,2],
          [1,2],
          [3,4],
          [3,4],
          [0],
          [0]]
docs_c = [[2,1],
          [4,1],
          [3,1],
          [4,2],
          [5],
          [4]]
phi, theta = lda.fit(corpus)
for k in range(n_topics):
    print "topic:", k
    indexes = np.argsort(phi[k])
    for word in indexes[::-1][:10]:
        print dictionary[word]
    print ""


