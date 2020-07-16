from scipy import stats
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform


def randomized_grid_search(
    df,
    pipeline,
    objective_metric_name='roc_auc',
    n_iter_search=500
    ):
    scoring = {
        'accuracy': metrics.make_scorer(metrics.accuracy_score),
        'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
        'precision': metrics.make_scorer(metrics.average_precision_score),
        'recall': metrics.make_scorer(metrics.recall_score)}

    X = df['clean_text']
    y = df['target']

    hyperparam_grid = {
        "vectorizer__ngram_range": [(1, 1), (1, 2), (2, 3)],
        "vectorizer__min_df": stats.randint(1, 3),
        "vectorizer__max_df": stats.uniform(.95, .3),
        "vectorizer__sublinear_tf": [True, False],
        "vectorizer__analyzer": ["char"],
        "clf__penalty": ['l2', 'l1', 'elasticnet'],
        "clf__loss": ['log', 'modified_huber'],
        "clf__tol": loguniform(1e-4, 1e-3),
        "clf__epsilon": loguniform(1e-3, 1e-1),
        "clf__n_iter_no_change": [10, 15, 20],
        "clf__max_iter": stats.randint(1000, 3000)
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=hyperparam_grid,
        scoring=scoring,
        refit=objective_metric_name,
        n_iter=n_iter_search,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=123
    )

    random_search.fit(X, y)
    best_estimator = random_search.best_estimator_

    return best_estimator
