import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin


# Pandas one hot encoder
class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_values_for_col = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.unique_values_for_col[col] = X[col].unique().tolist()

    def transform(self, X):
        dataframes = []
        for col in X.columns:
            type = CategoricalDtype(categories=self.unique_values_for_col[col])
            dataframes.append(pd.get_dummies(pd.Series(X[col], dtype=type), prefix=col))
        return pd.concat(dataframes, axis=1)
