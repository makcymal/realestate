import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import TargetEncoder


class MyKNNImputer(BaseEstimator):
  def __init__(self, bool_cols: list[str]):
    self.knn = KNNImputer()
    self.bool_cols = bool_cols

  def fit(self, X: pd.DataFrame, *args, **kwargs):
    self.knn.fit(X)
    self.knn.set_output(transform="pandas")
    return self

  def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    X = self.knn.transform(X)
    for col in self.bool_cols:
      X[col] = np.round(X[col])
    return X


class CatgImputer(BaseEstimator):
  def __init__(self, catg_cols: list[str]):
    self.catg_cols = catg_cols
    self.clf = {col: RandomForestClassifier() for col in self.catg_cols}

  def fit(self, X: pd.DataFrame, *args, **kwargs):
    lhs = X.drop(columns=self.catg_cols)
    for col in self.catg_cols:
      rhs = X[col].dropna()
      self.clf[col].fit(lhs.loc[rhs.index], rhs)
    return self

  def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    for col in self.catg_cols:
      to_impute = X[X[col].isna()].drop(
        columns=self.catg_cols, errors="ignore"
      )
      if to_impute.shape[0] == 0:
        continue
      X.loc[to_impute.index, col] = self.clf[col].predict(to_impute)
    return X
