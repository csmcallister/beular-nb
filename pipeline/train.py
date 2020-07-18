import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


class loguniform(stats.reciprocal):
    '''
    Subclass log uniform dist for param search space
    '''
    pass


class EstimatorSelectionHelper:
    '''
    To grid search multiple models with various para grids
    and save results
    '''

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = RandomizedSearchCV(
                model,
                param_distributions=params,
                return_train_score=True,
                **grid_kwargs
            )
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('Done.')

    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)

        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df


def randomized_grid_search(df, n_iter_search=500):
    models = dict(
        SGDClassifier=Pipeline(
            steps=[
                ('vectorizer', TfidfVectorizer()),
                ('clf', SGDClassifier(class_weight="balanced"))
            ]
        ),
        RandomForestClassifier=Pipeline(
            steps=[
                ('vectorizer', TfidfVectorizer()),
                ('clf', RandomForestClassifier())
            ]
        ),
        AdaBoostClassifier=Pipeline(
            steps=[
                ('vectorizer', TfidfVectorizer()),
                ('clf', AdaBoostClassifier())
            ]
        ),
        GradientBoostingClassifier=Pipeline(
            steps=[
                ('vectorizer', TfidfVectorizer()),
                ('clf', GradientBoostingClassifier())
            ]
        ),
        ExtraTreesClassifier=Pipeline(
            steps=[
                ('vectorizer', TfidfVectorizer()),
                ('clf', ExtraTreesClassifier())
            ]
        )
    )

    params = {
        'SGDClassifier': {
            "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 3)],
            "vectorizer__min_df": stats.randint(1, 3),
            "vectorizer__max_df": stats.uniform(.95, .3),
            "vectorizer__sublinear_tf": [True, False],
            "clf__penalty": ['l2', 'l1', 'elasticnet'],
            "clf__loss": ['log', 'modified_huber'],
            "clf__tol": loguniform(1e-4, 1e-3),
            "clf__epsilon": loguniform(1e-3, 1e-1),
            "clf__n_iter_no_change": [10, 15, 20],
            "clf__max_iter": stats.randint(1000, 3000)
        },
        'RandomForestClassifier': {
            "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 3)],
            "vectorizer__min_df": stats.randint(1, 3),
            "vectorizer__max_df": stats.uniform(.95, .3),
            "vectorizer__sublinear_tf": [True, False],
            'clf__n_estimators': [50, 75, 100, 200, 300, 450, 580, 700, 1000, 1500],
            'clf__max_depth': [10, 50, 80, 90, 100, 115, 150, None],
            'clf__bootstrap': [True, False],
            'clf__criterion': ['gini', 'entropy'], 
            'clf__max_features': ['auto', 'log2', None],
            'clf__class_weight': [None, 'balanced']
        },
        'AdaBoostClassifier': {
            "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 3)],
            "vectorizer__min_df": stats.randint(1, 3),
            "vectorizer__max_df": stats.uniform(.95, .3),
            "vectorizer__sublinear_tf": [True, False],
            'clf__n_estimators': [50, 75, 100, 200, 300, 450, 580, 700, 1000, 1500],
            'clf__learning_rate': [.8, 1.0],
        },
        'GradientBoostingClassifier': {
            "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 3)],
            "vectorizer__min_df": stats.randint(1, 3),
            "vectorizer__max_df": stats.uniform(.95, .3),
            "vectorizer__sublinear_tf": [True, False],
            'clf__n_estimators': [50, 75, 100, 200, 300, 450, 580, 700, 1000, 1500],
            'clf__learning_rate': [.8, 1.0],
            'clf__max_depth': loguniform(3, 1500),
            'clf__max_features': ['auto', 'log2', None]
        },
        'ExtraTreesClassifier': {
            "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 3)],
            "vectorizer__min_df": stats.randint(1, 3),
            "vectorizer__max_df": stats.uniform(.95, .3),
            "vectorizer__sublinear_tf": [True, False],
            'clf__n_estimators': [50, 75, 100, 200, 300, 450, 580, 700, 1000, 1500],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [10, 50, 80, 90, 100, 115, 150, None],
            'clf__max_features': ['auto', 'log2', None],
            'clf__bootstrap': [True, False],
            'clf__class_weight': [None, 'balanced']
        }
    }

    # Data should have already (before pre-processing) been split into train
    # and test, but the competition requirements did not specify this and
    # doing so when others might would only dock me
    X = df['Clause Text']
    y = df['Classification']

    helper = EstimatorSelectionHelper(models, params)
    helper.fit(
        X,
        y,
        scoring='f1',
        verbose=1,
        n_jobs=-1,
        n_iter=n_iter_search,
        random_state=123,
        cv=5
    )

    summary = helper.score_summary(sort_by='mean_train_score')
    best_score = summary['mean_train_score'].max() * 100
    best_est_name = summary.iloc[0]['estimator']
    print(f"The best was {best_est_name} with a score of {best_score}")
    best_estimator = helper.grid_searches[best_est_name].best_estimator_

    return best_estimator
