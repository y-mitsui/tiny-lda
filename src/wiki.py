import MySQLdb
from parallel_vb import LDA
#from vb import LDA
import numpy as np
from gensim import corpora, models, similarities
import re
import nltk
import time
import pickle
#from lda_wrap import LDAWrap

np.random.seed(12345)

if False:
    connection = MySQLdb.connect(db="nlp_dataset",user="root",passwd="password")
    connection.set_character_set('utf8')
    cursor = connection.cursor()
    cursor.execute("SELECT content FROM wikipedia_en ORDER BY id LIMIT 40000")
    documents = []
    print "reading"
    for search_content, in cursor.fetchall():
        documents.append(search_content)

    symbols = "<>^[]!-/:@-`{-~,;*+|=.#$%&?\\"
    numeric = re.compile("^[0-9\.%:,]+$")
    numeric2 = re.compile("[0-9]")
    double_quote = re.compile("\"([A-Za-z0-9]+)\"")
    single_quote = re.compile("'([A-Za-z0-9]+)'")
    kakko_quote = re.compile("\(([A-Za-z0-9]+)\)")

    stoplist = {}
    with open("stop_words.txt", "r") as fh:
        for line in fh:
            stoplist[line.strip()] = True
            
    print "split"
    texts = []
    word_counter = {}
    for document in documents:
        words = []
        for word in document.lower().split():
            if word not in stoplist and word[0] not in symbols \
                and numeric.match(word) is None and numeric2.search(word) is None:
                if word[-1] in ",.!?":
                    word = word[:-1]
                result = double_quote.match(word)
                if result is not None:
                    word = result.group(1)
                result = kakko_quote.match(word)
                if result is not None:
                    word = result.group(1)
                
                if word[0] in "\"\'(" or word[-1] in "\"\'(":
                    continue
                if word[:2] == "0x":
                    continue
                    
                words.append(word)
                if word not in word_counter:
                    word_counter[word] = 0
                word_counter[word] += 1
        texts.append(words)

    new_texts = []
    for text in texts:
        row = []
        for word, tag in nltk.pos_tag(text):
            if tag in ['NN', 'NNS', 'NP', 'NPS']:
                row.append(word)
        new_texts.append(row)
    texts = new_texts
    with open("texts.p", "w") as fh:
        pickle.dump((texts, word_counter), fh)
else:
    with open("texts.p", "r") as fh:
        texts, word_counter = pickle.load(fh)

print "word"
texts = [[word for word in text if word_counter[word] > 10]
            for text in texts]
texts = filter(lambda words: len(words) > 10, texts)
            
print "dictionary"
if False:
    dictionary = corpora.Dictionary()
    dictionary.load('/tmp/deerwester.dict')
else:
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text('/tmp/wiki_en.txt')
    
print "bow"
corpus = [dictionary.doc2bow(text) for text in texts]
n_topics = 20
lda = LDA(n_topics, 100, alpha=0.1, beta=0.1)
t1 = time.time()
phi, theta = lda.fit_transform(corpus)
print("train time %.1fsec"%(time.time() - t1))
for k in range(n_topics):
    print("topic:", k)
    indexes = np.argsort(phi[k])
    for word in indexes[::-1][:30]:
        print(dictionary[word])
    print("")
    
