import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.base import BaseEstimator


class Envelope(BaseEstimator):

  def fit(self, *args, **kwargs):
    return self

  # inheritors always return pandas outputs
  def set_output(self, *args, **kwargs):
    pass


class MultiColLabelEncoder:
  
  def 
