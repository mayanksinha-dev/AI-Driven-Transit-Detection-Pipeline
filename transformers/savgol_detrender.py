from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter

class SavGolDetrender(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, data):

        output = []

        for d in data:

            trend = savgol_filter(
                d["flux"],
                101,
                3
            )

            d["flux"] = d["flux"] / trend

            output.append(d)

        return output