from utils.utils import Utils
from nlp_utils.tokenizer import Tokenizer


class ListCorpusReader:
    """
    Inputs a list of strings and generates a processed corpus consisting of a nested list of tokenized documents.
    """

    def __init__(self, raw_corpus):
        self.corpus = self.load_corpus(raw_corpus)

    @staticmethod
    def load_corpus(raw_corpus):
        """
        Loads raw corpus.
        """
        return Utils.pipeline(raw_corpus, Tokenizer.nltk_regexp_tokenize)

    def __str__(self):
        return str(self.corpus)
