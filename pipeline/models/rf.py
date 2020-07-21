from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', RandomForestClassifier())]
)

params = {
    "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)],
    "vectorizer__min_df": stats.randint(1, 3),
    "vectorizer__max_df": stats.uniform(.95, .3),
    "vectorizer__sublinear_tf": [True, False],
    'clf__n_estimators': stats.randint(100, 300),
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_features': ['auto', 'log2', None],
    'clf__max_depth': stats.randint(10, 150),
    'clf__class_weight': [None, 'balanced'],
    'clf__min_samples_split': stats.reciprocal(.0001, .2),
    'clf__min_samples_leaf': stats.reciprocal(.0001, .2)
}
