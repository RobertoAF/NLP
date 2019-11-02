from collections import defaultdict
import math

import numpy as np
from sklearn import preprocessing

from nlp_utils.list_corpus_reader import ListCorpusReader


class MinTfidfVectorizer:
    def __init__(self):
        self.processed_corpus = None
        self.unique_word_count = defaultdict(int)
        self.idf = defaultdict(float)

    @staticmethod
    def doc_tf_counter(doc):
        counter = defaultdict(float)
        for word in doc:
            counter[word] += 1.
        return counter

    def tf(self, doc):
        tf_count = self.doc_tf_counter(doc)
        doc_length = len(doc)
        for word in tf_count:
            tf_count[word] = tf_count[word] / doc_length
            # count the number of documents in which this unique word appears
            self.unique_word_count[word] += 1
        return tf_count

    def get_tf(self):
        return [self.tf(doc) for doc in self.processed_corpus]

    def get_idf(self):
        corpus_length = len(self.processed_corpus)
        for word in self.unique_word_count:
            self.idf[word] = math.log((corpus_length / self.unique_word_count[word] + 1.))
        return self.idf

    def tf_idf(self, doc_tf):
        tf_idf = defaultdict(float)
        for word in doc_tf:
            tf_idf[word] = doc_tf[word] * self.idf[word]
        return tf_idf

    def get_tf_idf(self):
        tf = self.get_tf()
        self.idf = self.get_idf()
        return [self.tf_idf(doc_tf) for doc_tf in tf]

    def tf_idf_vector(self, tf_idf_doc):
        tf_idf_vector = np.zeros(len(self.unique_word_count))
        for i, word in enumerate(self.unique_word_count):
            if word in tf_idf_doc:
                tf_idf_vector[i] = tf_idf_doc[word]
        return tf_idf_vector

    def get_tf_idf_vectors(self):
        tf_idf = self.get_tf_idf()
        return np.array([self.tf_idf_vector(tf_idf_doc) for tf_idf_doc in tf_idf])

    def fit_transform(self, raw_corpus):
        self.processed_corpus = ListCorpusReader(raw_corpus).corpus
        tf_idf_matrix = self.get_tf_idf_vectors()
        # use L-2 normalization for the tf-idf matrix output
        normalized_tf_idf_matrix = preprocessing.normalize(tf_idf_matrix, norm='l2')
        # return normalized_tf_idf matrix
        return normalized_tf_idf_matrix
