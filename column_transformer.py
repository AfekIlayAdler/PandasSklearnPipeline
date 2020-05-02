import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# selects subsets of columns from pandas dataframe
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


# checks for every value in the column id it belongs to a set of values
class ColumnInOptions(BaseEstimator, TransformerMixin):
    def __init__(self, options):
        self.values = options

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.Series)
        return X.where(X.isin(self.values), np.nan)


# transforms values in a column
class ColumnApplyFunc(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.Series)
        return self.func(X)


class ColumnExist(ColumnApplyFunc):
    def __init__(self):
        super().__init__(pd.Series.notna)


# changes dtype of a column
class ColumnAsType(BaseEstimator, TransformerMixin):
    def __init__(self, category):
        self.category = category

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.Series)
        return X.astype(self.category)
