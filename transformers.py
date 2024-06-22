import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option("future.no_silent_downcasting", True)


# remove spaces, replace comma with dot
def clean_float_object(obj: str) -> str:
  if pd.isna(obj):
    return "nan"

  obj = list(obj)
  dots = 0

  ins = 0
  for i in range(len(obj)):
    if obj[i] == ",":
      obj[i] = "."
    if obj[i] == ".":
      dots += 1
    if not obj[i].isspace():
      obj[ins] = obj[i]
      ins += 1

  if dots > 1:
    ins = 0
    for i in range(len(obj)):
      if obj[i] == "." and dots > 1:
        dots -= 1
      else:
        obj[ins] = obj[i]
        ins += 1

  return "".join(obj[:ins])


# remove spaces, commas and dots
def clean_int_object(obj: str) -> str:
  if pd.isna(obj):
    return "nan"
  obj = list(obj)
  ins = 0
  for i in range(len(obj)):
    if not (obj[i] in [",", ".", "_"] or obj[i].isspace()):
      obj[ins] = obj[i]
      ins += 1
  return "".join(obj[:ins])


# the most basic transformer
class OptimusPrime(BaseEstimator):

  def fit(self, *args, **kwargs):
    return self

  # inheritors return pandas outputs
  def set_output(self, *args, **kwargs):
    pass


class IntTransformer(OptimusPrime):

  def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    return X.map(self._into_int).astype("Int32")

  def _into_int(self, value) -> int:
    # convert string to float (float can be NaN)
    if isinstance(value, str):
      value = float(clean_int_object(value))
    # if NaN, return pd.NA that is integer compatible
    if pd.isna(value):
      return pd.NA
    return int(value)


class FloatTransformer(OptimusPrime):

  def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    return X.map(self._into_float).astype("Float32")

  def _into_float(self, value) -> float:
    # convert string to float (float can be NaN)
    if isinstance(value, str):
      value = float(clean_float_object(value))
    if pd.isna(value):
      return np.nan
    return value


class DatetimeTransformer(OptimusPrime):

  def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    # cannot work with the whole dataframe
    for col in X.columns:
      X[col] = pd.to_datetime(X[col], dayfirst=True, format="mixed")
    return X


class BooleanTransformer(OptimusPrime):

  MAP = {"нет": 0, "да": 1, "неизвестно": pd.NA}

  def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    return X.map(
      lambda value: self.__class__.MAP[value.lower().strip()]
    ).astype("Int32")
