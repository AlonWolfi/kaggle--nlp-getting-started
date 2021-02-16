from sklearn.model_selection import StratifiedShuffleSplit

def cross_val_predict_proba(clf, X, y):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    sss.get_n_splits(X, y)
    y_pred = y.copy()
    for train_index, test_index in sss.split(X, y):
        xx_train, xx_test = X[train_index], X[test_index]
        yy_train, yy_test = y[train_index], y[test_index]
        clf.fit(xx_train, yy_train)
        y_pred[test_index] = clf.predict_proba(xx_test)[:,1]
    return y_pred