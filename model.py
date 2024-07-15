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


class PassthroughTransformer(BaseEstimator):
  def fit(self, X: pd.DataFrame, *args, **kwargs):
    return self

  def transform(self, X: pd.Series) -> pd.Series:
    return X

  def inverse_transform(self, X: pd.Series) -> pd.Series:
    return X


class LogarithmicTransformer(BaseEstimator):
  def fit(self, X: pd.DataFrame, *args, **kwargs):
    return self

  def transform(self, X: pd.Series) -> pd.Series:
    return np.log(X)

  def inverse_transform(self, X: pd.Series) -> pd.Series:
    return np.exp(X)


class NoiseAdder(BaseEstimator):
  def fit(self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs):
    return self

  def transform(
    self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs
  ) -> pd.DataFrame:
    rng = np.random.default_rng(424)
    noised = X.copy()
    for col in X.columns:
      std = X[col].std()
      noised[col] += rng.normal(loc=0, scale=std, size=noised[col].shape)

    return pd.concat([X, noised], ignore_index=True), pd.concat(
      [y, y], ignore_index=True
    )
