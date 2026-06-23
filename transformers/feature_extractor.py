from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew, kurtosis
import pandas as pd
import numpy as np

class FeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, data):

        rows = []

        for d in data:

            flux = d["flux"]

            q25 = np.percentile(
                flux,
                25
            )

            q75 = np.percentile(
                flux,
                75
            )

            rows.append({

                "mean_flux":
                    np.mean(flux),

                "std_flux":
                    np.std(flux),

                "amplitude":
                    np.max(flux) -
                    np.min(flux),

                "skewness":
                    skew(flux),

                "kurtosis":
                    kurtosis(flux),

                "mad":
                    np.median(
                        np.abs(
                            flux -
                            np.median(flux)
                        )
                    ),

                "flux_q25":
                    q25,

                "flux_q75":
                    q75,

                "iqr":
                    q75 - q25,

                "bls_period":
                    d["period"],

                "bls_duration":
                    d["duration"],

                "bls_depth":
                    d["depth"],

                "bls_transit_time":
                    d["transit_time"],

                "transit_snr":
                    d["transit_snr"],

                "bls_power":
                    d["bls_power"]

            })

        return pd.DataFrame(rows)