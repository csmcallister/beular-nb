from collections import defaultdict
import os
import unicodedata

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy import stats
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


class MeanEmbeddingVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array(
            [np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X]
        )
            

class TfidfEmbeddingVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, word2vec, **kwargs):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))
        self.kwargs = kwargs

    def fit(self, X, y):
        tfidf = TfidfVectorizer(**self.kwargs)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        
        
class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)
    
    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token)
        ]
    
    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            return self.normalize(document)


embeddings_dict = {}
with open("glove.6B.100d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


pipe = Pipeline([
    ('vectorizer', MeanEmbeddingVectorizer(embeddings_dict)),
    ('clf', SGDClassifier(class_weight="balanced"))]
)

params = {
    #"vectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)],
    #"vectorizer__min_df": stats.randint(1, 3),
    #"vectorizer__max_df": stats.uniform(.95, .3),
    #"vectorizer__sublinear_tf": [True, False],
    "clf__penalty": ['l2', 'l1', 'elasticnet'],
    "clf__loss": ['log', 'modified_huber'],
    "clf__tol": stats.reciprocal(1e-4, 1e-3),
    "clf__epsilon": stats.reciprocal(1e-3, 1e-1),
    "clf__n_iter_no_change": [10, 15, 20],
    "clf__max_iter": stats.randint(1000, 3000)
}
