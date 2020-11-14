import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict


class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators: list, final_estimator: BaseEstimator):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def fit(self, X, y):
        if type(X) == pd.DataFrame:
            X = X.values

        for name, clf in self.estimators:
            clf.fit(X, y)
        self.classes_ = self.estimators[0][1].classes_
        X_ensemble = pd.DataFrame(
            {name: cross_val_predict(clf, X, y, method='predict_proba')[:, 1] for name, clf in self.estimators})
        self.final_estimator.fit(X_ensemble, y)

    def predict_proba(self, X):
        if type(X) == pd.DataFrame:
            X = X.values
        X_ensemble = pd.DataFrame({name: clf.predict_proba(X)[:, 1] for name, clf in self.estimators})
        return self.final_estimator.predict_proba(X_ensemble)

    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5
