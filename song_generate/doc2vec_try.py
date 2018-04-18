from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

doc1 = TaggedDocument(words=['some', 'words', 'here'], tags=['sent_1'])
listwords = ['some', 'words', 'here']
sentences = [doc1]
model = Doc2Vec(size=100, window=300, min_count=10, workers=4)
model.build_vocab(sentences)

for i in range(10):
    model.train(sentences)


model.infer_vector(listwords)