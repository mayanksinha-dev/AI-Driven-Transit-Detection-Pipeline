from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MADFilter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, data):

        output = []

        for d in data:

            flux = d["flux"]

            med = np.median(flux)

            mad = np.median(
                np.abs(flux - med)
            )

            mask = np.abs(
                flux - med
            ) < 5 * mad

            d["time"] = d["time"][mask]
            d["flux"] = d["flux"][mask]

            output.append(d)

        return output