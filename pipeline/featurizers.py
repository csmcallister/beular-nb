import re

import contractions
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
        self.no_nonsense_re = re.compile(r'^[a-zA-Z]+$')

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        elif treebank_tag == 'PRP':
            return wordnet.ADJ_SAT
        elif treebank_tag == 'MD':
            return 'n'
        else:
            return ''

    def fit(self, X, y=None):
        return self

    def _preprocessing(self, text):
        """
        Lemmatizes lowercased alpha-only substrings to be b/w 3 and 17 chars.

        Parameters:
            doc (text): the text of a clause

        Returns:
            words (str): a space-delimited lower-case alpha-only str
        """

        text = contractions.fix(text, slang=False)
        lemmatizer = WordNetLemmatizer()
        tagged_tokens = nltk.pos_tag(word_tokenize(text))
        words = ''
        for token, pos in tagged_tokens:
            wordnet_pos = TextPreprocessor.get_wordnet_pos(pos)
            if wordnet_pos:
                lemma = lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
                if re.match(self.no_nonsense_re, lemma):
                    words += f' {lemma}'

        return words.strip()

    def transform(self, X, y=None):
        X = X.apply(self._preprocessing)
        return X
