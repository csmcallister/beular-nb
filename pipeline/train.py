from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import metrics

from models.sgd import pipe, params


def randomized_grid_search(X, y, n_iter=1, score='precision'):

    scoring = {
        'accuracy': metrics.make_scorer(metrics.accuracy_score),
        'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
        'precision': metrics.make_scorer(metrics.average_precision_score),
        'fbeta': metrics.make_scorer(metrics.fbeta_score, beta=1),
        'recall': metrics.make_scorer(metrics.recall_score)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=1020
    )

    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        scoring=scoring,
        refit=score,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=1020
    )
    random_search.fit(X_train, y_train)

    y_prob = random_search.predict_proba(X_test)
    positive_class_col = list(random_search.classes_).index(1)
    y_score = y_prob[:, positive_class_col]

    y_pred = []
    for comp_prob, _ in y_prob.tolist():
        y_pred.append(1 if comp_prob > .5 else 0)
    f1 = metrics.f1_score(y_test, y_pred)
    bs = metrics.brier_score_loss(y_test, y_score)
    average_precision = metrics.average_precision_score(y_test, y_score)
    acc = metrics.accuracy_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    precisions, recalls, _ = metrics.precision_recall_curve(y_test, y_score)
    auc = metrics.auc(recalls, precisions)
    recall = metrics.recall_score(y_test, y_pred)
    best_score = random_search.best_score_

    print("*"*80)
    print("\tRecall:  {0:.2f}".format(recall))
    print("\tAccuracy:  {0:.2f}".format(acc))
    print("\tROC-AUC:  {0:.2f}".format(roc_auc))
    print("\tF1:  {0:.2f}".format(f1))
    print("\tAverage Precision:  {0:.2f}".format(average_precision))
    print("\tBrier Score:  {0:.2f}".format(bs))
    print("\tPrecision-Recall AUC:  {0:.2f}".format(auc))
    print("\tBest mean cross-validated score: {0:.2f}".format(best_score))

    print("-"*80)
    print("Classification Report:")
    print(metrics.classification_report(
        y_test,
        y_pred,
        target_names=['Compliant', 'NonCompliant'])
    )

    return random_search.best_estimator_
