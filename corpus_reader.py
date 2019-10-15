from functools import reduce

from nltk.tokenize.regexp import regexp_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize


class CorpusReader:
    """
    Inputs a raw corpus and generates a processed corpus consisting of a nested list of tokenized documents.
    """

    def __init__(self, raw_corpus):
        self.corpus = self.load_corpus(raw_corpus)

    @staticmethod
    def pipeline(x, *funcs):
        """
        A functional approach for executing a sequence of methods applied to the same argument.
        """
        return reduce(lambda x, f: f(x), funcs, x)

    def load_corpus(self, raw_corpus):
        """
        Loads raw corpus.
        """
        return self.pipeline(raw_corpus, self.nltk_regexp_tokenize)

    @staticmethod
    def nltk_regexp_tokenize(raw_corpus):
        re_pattern = '\w+|\$[\d\.]+|\S+'
        return [regexp_tokenize(doc, re_pattern) for doc in raw_corpus]

    @staticmethod
    def nltk_tokenize_tweets(raw_corpus):
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        return [tokenizer.tokenize(doc) for doc in raw_corpus]

    @staticmethod
    def nltk_word_tokenize(raw_corpus):
        return [word_tokenize(doc) for doc in raw_corpus]

    def __str__(self):
        return str(self.corpus)
