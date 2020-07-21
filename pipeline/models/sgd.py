from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', SGDClassifier(class_weight="balanced"))]
)

params = {
    "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)],
    "vectorizer__min_df": stats.randint(1, 3),
    "vectorizer__max_df": stats.uniform(.95, .3),
    "vectorizer__sublinear_tf": [True, False],
    "clf__penalty": ['l2', 'l1', 'elasticnet'],
    "clf__loss": ['log', 'modified_huber'],
    "clf__tol": stats.reciprocal(1e-4, 1e-3),
    "clf__epsilon": stats.reciprocal(1e-3, 1e-1),
    "clf__n_iter_no_change": [10, 15, 20],
    "clf__max_iter": stats.randint(1000, 3000)
}
