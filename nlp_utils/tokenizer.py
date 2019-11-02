from nltk.tokenize.regexp import regexp_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize


class Tokenizer:
    @staticmethod
    def nltk_regexp_tokenize(raw_corpus):
        # regular expression pattern includes punctuation
        re_pattern = '\w+|\$[\d\.]+|\S+'
        return [regexp_tokenize(doc, re_pattern) for doc in raw_corpus]

    @staticmethod
    def nltk_tokenize_tweets(raw_corpus):
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        return [tokenizer.tokenize(doc) for doc in raw_corpus]

    @staticmethod
    def nltk_word_tokenize(raw_corpus):
        return [word_tokenize(doc) for doc in raw_corpus]