import math
from multiprocessing import cpu_count

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

from models.sgd import pipe, params


N_JOBS = math.ceil(cpu_count() * .8)


def randomized_grid_search(X_train, y_train, n_iter=1, score='precision'):

    scoring = {
        'accuracy': metrics.make_scorer(metrics.accuracy_score),
        'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
        'precision': metrics.make_scorer(metrics.average_precision_score),
        'fbeta': metrics.make_scorer(metrics.fbeta_score, beta=1),
        'recall': metrics.make_scorer(metrics.recall_score)
    }

    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        scoring=scoring,
        refit=score,
        n_iter=n_iter,
        n_jobs=N_JOBS,
        cv=5,
        verbose=1,
        random_state=123
    )

    random_search.fit(X_train, y_train)
    
    best_score = random_search.best_score_
    best_params = random_search.best_params_

    print("*"*80)
    print("Best mean cross-validated score: {0:.2f}".format(best_score)) 
    print("BEST PARAMS")
    print(best_params)

    return random_search.best_estimator_
