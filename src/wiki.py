import MySQLdb
from svb import LDA
import numpy as np
from gensim import corpora, models, similarities
import re

np.random.seed(12345)
connection = MySQLdb.connect(db="nlp_dataset",user="root",passwd="password")
connection.set_character_set('utf8')
cursor = connection.cursor()
cursor.execute("SELECT content FROM wikipedia_en ORDER BY id LIMIT 50000")
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
n_topics = 100
lda = LDA(n_topics, 5000, step_size=0.005)
phi, theta = lda.fit(corpus)
for k in range(n_topics):
    print("topic:", k)
    indexes = np.argsort(phi[k])
    for word in indexes[::-1][:30]:
        print(dictionary[word])
    print("")
    
