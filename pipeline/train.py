import json
import math
from multiprocessing import cpu_count

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

from models.gbc import pipe, params


N_JOBS = math.ceil(cpu_count() * .8)


def randomized_grid_search(X, y, n_iter=1, score='precision'):

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

    random_search.fit(X, y)

    y_prob = random_search.predict_proba(X)
    y_pred = random_search.predict(X)
    positive_class_col = list(random_search.classes_).index(1)
    y_score = y_prob[:, positive_class_col]

    f1 = metrics.f1_score(y, y_pred)
    bs = metrics.brier_score_loss(y, y_score)
    average_precision = metrics.average_precision_score(y, y_score)
    acc = metrics.accuracy_score(y, y_pred)
    roc_auc = metrics.roc_auc_score(y, y_pred)
    precisions, recalls, _ = metrics.precision_recall_curve(y, y_score)
    auc = metrics.auc(recalls, precisions)
    recall = metrics.recall_score(y, y_pred)
    best_score = random_search.best_score_
    best_params = random_search.best_params_

    print("*"*80)
    print("Recall:  {0:.2f}".format(recall))
    print("Accuracy:  {0:.2f}".format(acc))
    print("ROC-AUC:  {0:.2f}".format(roc_auc))
    print("F1:  {0:.2f}".format(f1))
    print("Average Precision:  {0:.2f}".format(average_precision))
    print("Brier Score:  {0:.2f}".format(bs))
    print("Precision-Recall AUC:  {0:.2f}".format(auc))
    print("Best mean cross-validated score: {0:.2f}".format(best_score)) 

    print("-"*80)
    print("Classification Report:")
    print(metrics.classification_report(y, y_pred, target_names=['Compliant', 'NonCompliant']))
    print("BEST PARAMS")
    print(best_params)

    return random_search.best_estimator_
