from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


pipe = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('clf', GradientBoostingClassifier())]
)

params = {
    "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)],
    "vectorizer__min_df": stats.randint(1, 3),
    "vectorizer__max_df": stats.uniform(.95, .3),
    "vectorizer__sublinear_tf": [True, False],
    'clf__n_estimators': stats.randint(100, 300),
    'clf__min_samples_split': stats.reciprocal(.0001, .2),
    'clf__min_samples_leaf': stats.reciprocal(.0001, .2),
    'clf__learning_rate': stats.reciprocal(.001, 1.0),
    'clf__max_depth': stats.randint(3, 150),
    'clf__max_features': ['auto', 'log2', None]
}
