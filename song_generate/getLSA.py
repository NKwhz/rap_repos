from gensim import corpora, similarities, models
import re
import string
import os
class LSA:

    def __init__(self, corpus_filename=None,dictionary='lsa/dictionary.txt',corpus='lsa/corpuse.mm',tfidf_model='lsa/tfidf_model.txt',lsi_model='lsa/lsi_model.txt'):
        '''

        :param corpus_filename: the filename of traning data
        :param dictionary: had store dictionary name
        :param tfidf_model: had store tfidf_model name
        :param lsi_model: had store lsi_model name

        you have two choices
        1. provide corpus_filename
        2. provide dictionary and tfidf_model and lsi_model
        '''
        self.delset = string.punctuation
        if corpus_filename is None:#choices 2
            self.dictionary=corpora.Dictionary.load(dictionary)
            self.corpus=corpora.MmCorpus(corpus)
            self.tfidf_model=models.TfidfModel.load(tfidf_model)
            self.lsi_model=models.LsiModel.load(lsi_model, mmap='r')


        else:#choices 1  need get dictionary tfidf_model lsi_model
            self.build(filename=corpus_filename)






    def build(self,filename=None):
        '''
                input: filename which contain candidate lines
                aim :1. Remove the content of the brackets {} []
                    2.Replaced with lowercase
                    3.cut words
                    4.Remove the stop words

                    5.build dictionary        no the words of the frequence is  below 3

                    6.build tfidf_model
                    7.build lsi_model


        '''
        self.get_stopword()
        f=open(filename)
        #
        raw_documents = f.readlines()
        corpora_documents = []
        #
        for item_text in raw_documents:
            item_text=item_text.strip()


            # #Remove the content of the brackets {} []
            pattern = re.compile(r'(\{)(.*)(\})|(\[)(.*)(\])')

            item_text=pattern.sub(r'', item_text)
            item_text = item_text.translate(None,self.delset)
            #Replaced with lowercase
            item_text=item_text.lower()
            #cut words
            item_seg = list(item_text.split())
            #Remove the stop words
            for it in item_seg:
                if it in self.stopwords:
                    item_seg.remove(it)

            corpora_documents.append(item_seg)
        if not os.path.exists('lsa'):
            os.makedirs('lsa')
        dictionary = corpora.Dictionary(corpora_documents)
        dictionary.filter_extremes(no_below=3)  # no the words of the frequence is  below 3
        self.dictionary=dictionary
        dictionary.save('lsa/dictionary.txt')
        self.corpus = [dictionary.doc2bow(text) for text in corpora_documents]
        corpora.MmCorpus.serialize('lsa/corpuse.mm', self.corpus)
        self.tfidf_model = models.TfidfModel(self.corpus)
        self.tfidf_model.save('lsa/tfidf_model.txt')
        corpus_tfidf = self.tfidf_model[self.corpus]
        self.lsi_model = models.LsiModel(corpus_tfidf,num_topics=100)
        self.lsi_model.save('lsa/lsi_model.txt')




    def get_stopword(self):
        '''

        :return: list of stopwords
        '''
        self.stopwords = {}.fromkeys([line.rstrip() for line in open('stopwords.txt')])

    def get_train_build(self,train_filename,train_textlist):
        '''

        :param train_filename: train data filename (about 300 lines lyrics)
        :return: lsa value of train data
        '''
        self.get_stopword()
        if  not train_textlist is not None:

            f = open(train_filename)
            # print '00'
            #
            raw_documents = f.readlines()
        else :
            raw_documents=train_textlist
            # print '00l'
        corpora_documents = []
        #
        for item_text in raw_documents:
            item_text = item_text.strip()
            # # Remove the content of the brackets {} []
            pattern = re.compile(r'(\{)(.*)(\})|(\[)(.*)(\])')
            item_text = pattern.sub(r'', item_text)
            item_text = item_text.translate(None,self.delset)
            # Replaced with lowercase
            item_text = item_text.lower()
            # cut words
            item_seg = list(item_text.strip().split())
            # Remove the stop words
            for it in item_seg:
                if it in self.stopwords:
                    item_seg.remove(it)

            corpora_documents.append(item_seg)
        corpus = [self.dictionary.doc2bow(text) for text in corpora_documents]
        test_corpus_tfidf = self.tfidf_model[corpus]
        corpus_lsi = self.lsi_model[test_corpus_tfidf]
        return corpus_lsi

    def get_test_build(self,text):
        item_text = text.strip()
        # Remove the content of the brackets {} []
        pattern = re.compile(r'(\{)(.*)(\})|(\[)(.*)(\])')
        item_text = pattern.sub(r'', item_text)
        item_text = item_text.translate(None,self.delset)
        # # Replaced with lowercase
        item_text = item_text.lower()
        # cut words
        item_seg = list(item_text.strip().split())
        test_corpus = self.dictionary.doc2bow(item_seg)  #
        test_corpus_tfidf = self.tfidf_model[test_corpus] #
        test_corpus_lsi= self.lsi_model[test_corpus_tfidf]# 3
        return test_corpus_lsi

    def get_simli_trains_test(self,train_filename='train.txt',train_textlist=None,line="Growin up it's gettin a little better"):
        train=self.get_train_build(train_filename,train_textlist)
        test=self.get_test_build(line)
        similarity_lsi = similarities.Similarity('Similarity-LSI-index', train, num_features=400)
        return similarity_lsi[test]
# LSA('lsa/lsa_train_cop')
