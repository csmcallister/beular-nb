import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


class OptionalTruncatedSVD(TruncatedSVD):
    '''Subclass to make optional'''

    def __init__(
        self,
        passthrough=False,
        n_components=2,
        algorithm="randomized",
        n_iter=5,
        random_state=None,
        tol=0.
    ):
        self.passthrough = passthrough
        super().__init__(n_components, algorithm, n_iter, random_state, tol)

    def fit(self, X, y=None):
        if self.passthrough:
            return self
        else:
            return super().fit(X, y)

    def fit_transform(self, X, y=None):
        if self.passthrough:
            return X
        else:
            return super().fit_transform(X, y)

    def transform(self, X):
        if self.passthrough:
            return X
        else:
            return super().transform(X)


def lemmatizer(text):
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
    tagged_tokens = nltk.pos_tag(word_tokenize(text))
    lemmas = []
    for token, pos in tagged_tokens:
        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos:
            lemmas.append(WordNetLemmatizer().lemmatize(
                token.lower(),
                pos=wordnet_pos)
            )
    return lemmas


def stemmer(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


def nlkt_tokenize(text):
    return nltk.word_tokenize(text)


pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lsa', OptionalTruncatedSVD()),
    ('clf', SGDClassifier(class_weight="balanced"))
])


params = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (2, 2)],
    "tfidf__min_df": stats.randint(1, 3),
    "tfidf__max_df": stats.uniform(.95, .3),
    "tfidf__sublinear_tf": [True, False],
    "tfidf__tokenizer": [None, stemmer, lemmatizer, nlkt_tokenize],
    "lsa__passthrough": [True, False, True, True, True, True, True],
    "lsa__n_components": stats.randint(100, 3000),
    "clf__penalty": ['l2', 'l1', 'elasticnet'],
    "clf__loss": ['log', 'modified_huber'],
    "clf__tol": stats.reciprocal(1e-4, 1e-3),
    "clf__epsilon": stats.reciprocal(1e-3, 1e-1),
    "clf__n_iter_no_change": [10, 15, 20],
    "clf__max_iter": stats.randint(1000, 1500)
}
