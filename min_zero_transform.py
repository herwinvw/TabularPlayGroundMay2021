from sklearn.base import BaseEstimator, TransformerMixin # type: ignore
from typing import List, Any

import pandas as pd
class MinZeroTransform(BaseEstimator, TransformerMixin):
    """
    Transforms X to X-X.min(), so that the minimum is shifted to 0.
    X.min() is learned in the fit.
    """
    def transform(self, df:pd.DataFrame, **transform_params:Any)->pd.DataFrame:
        return df - self.min
    
    def fit(self, X:pd.DataFrame, y:Any=None, **fit_params:Any):
        self.min = X.min()
        return self