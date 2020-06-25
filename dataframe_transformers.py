from typing import Dict, List

from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColAsCategory(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = X[col].astype(int).astype('category')
        return X


class CatToInt(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cat_to_ints = {}

    def fit(self, X, y=None):
        for col in X.columns:
            temp_df = X[[col]]
            temp_df['new_values'] = pd.factorize(temp_df[col])[0]
            temp_df = temp_df.drop_duplicates()
            self.cat_to_ints[col] = temp_df.set_index(col).to_dict()['new_values']
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = X[col].map(self.cat_to_ints[col])
            X[col] = X[col].astype('category')
        return X


class CatColsToMeanResponseValue(BaseEstimator, TransformerMixin):
    def __init__(self, p: float):
        self.cols_to_mpr: Dict[str:Dict[str, float]] = {}

    def fit(self, X, y):
        for col in X.columns:
            temp_df = pd.DataFrame([X[col], y]).T
            self.cols_to_mpr[col] = temp_df.groupby(col).mean().to_dict()[y.name]
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        for col in X.columns:
            X[col] = X[col].map(self.cols_to_mpr[col])
        return X


class NanColumnsRemover(BaseEstimator, TransformerMixin):
    # removes columns with more than p% Nan
    def __init__(self, p: float):
        self.p = p
        self.columns_to_remove = None

    def fit(self, X, y=None):
        cols_nan_p = X.isna().sum() / X.shape[0]
        self.columns_to_remove = cols_nan_p[cols_nan_p > self.p].index.tolist()
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.drop(columns=self.columns_to_remove)


# selects columns that fits a given data type
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.dtype == 'bool':
            return X.select_dtypes(include=[self.dtype]) * 1
        return X.select_dtypes(include=[self.dtype])


class ObjectsColumnaAsType(BaseEstimator, TransformerMixin):
    def __init__(self):
        # transforms dype of an object column
        self.unique_values = {}  # key: colname, value: list of unique_items

    def fit(self, X, y=None):
        for col in X.select_dtypes(include='object').columns:
            self.unique_values[col] = X[col].dropna().unique()
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        # if at test time we encounter new values we map them for np.nan
        for col in self.unique_values.keys():
            X[col] = X[col].where(X[col].isin(self.unique_values[col]), np.nan)
            # binary/bool
            if len(self.unique_values[col]) == 2:
                X[col] = X[col].astype('bool')
            # category
            elif len(self.unique_values[col]) > 2:
                X[col] = X[col].astype('category')
        return X


class PandasImputer(BaseEstimator, TransformerMixin):
    # fill missing values
    fill_with = {'mean': np.mean, 'most_frequent': pd.Series.mode}

    def __init__(self, strategy):
        assert strategy in PandasImputer.fill_with.keys()
        self.strategy = strategy
        self.missing_values_map = {}
        self.imputer = PandasImputer.fill_with[strategy]

    def fit(self, X, y=None):
        for col in X.columns:

            if self.strategy == 'most_frequent':
                self.missing_values_map[col] = self.imputer(X[col]).values[0]
            else:
                self.missing_values_map[col] = self.imputer(X[col])
        return self

    def transform(self, X):
        for col in X.columns:
            try:
                X[col] = X[col].fillna(self.missing_values_map[col])
            except:
                try:
                    print(f"imputed diffrently for col:   {col}")
                    new_val = X[col][X[col].notna()].mode()[0]
                    X[col] = X[col].fillna(new_val)
                except:
                    print(f"imputed constant value for col: {col} shape of dataframe is {X.shape}")
                    X[col] = 1
        return X


class PandasStandardScaler(BaseEstimator, TransformerMixin):
    # weapper of StandardScaler()
    def __init__(self):
        self.transformer = None
        self.cols = None

    def fit(self, X, y=None):
        self.cols = X.columns.tolist()
        transformer = StandardScaler()
        self.transformer = transformer.fit(X)
        return self

    def transform(self, X):
        X.loc[:, self.cols] = self.transformer.transform(X[self.cols])
        return X


class IdentityTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X


class ColumnRemover(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.drop(columns=self.cols)
