from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os

class CurveExporter(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        save_path="curves.npz",
        target_len=2000
    ):
        self.save_path = save_path
        self.target_len = target_len

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        curves = []
        names = []

        for item in X:

            flux = item["flux"]

            if len(flux) > self.target_len:

                idx = np.linspace(
                    0,
                    len(flux)-1,
                    self.target_len
                ).astype(int)

                flux = flux[idx]

            else:

                flux = np.pad(
                    flux,
                    (0,
                     self.target_len-len(flux)),
                    mode="constant"
                )

            curves.append(flux)
            names.append(item["file"])

        np.savez_compressed(
            self.save_path,
            X=np.array(curves),
            files=np.array(names)
        )

        return X