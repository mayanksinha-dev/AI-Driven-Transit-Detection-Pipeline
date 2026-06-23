from sklearn.base import BaseEstimator, TransformerMixin
from astropy.timeseries import BoxLeastSquares
import numpy as np

class BLSSearch(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, data):

        output = []

        for d in data:

            try:

                time = np.asarray(d["time"])
                flux = np.asarray(d["flux"])

                if len(time) < 100:
                    continue

                bls = BoxLeastSquares(
                    time,
                    flux
                )

                periods = np.linspace(
                    0.5,
                    20,
                    3000
                )

                durations = np.linspace(
                    0.05,
                    0.30,
                    10
                )

                results = bls.power(
                    periods,
                    durations
                )

                idx = np.argmax(
                    results.power
                )

                d["period"] = float(
                    results.period[idx]
                )

                d["duration"] = float(
                    results.duration[idx]
                )

                d["depth"] = float(
                    results.depth[idx]
                )

                d["transit_time"] = float(
                    results.transit_time[idx]
                )

                d["bls_power"] = float(
                    results.power[idx]
                )

                std_flux = np.std(flux)

                d["transit_snr"] = (
                    d["depth"] / std_flux
                    if std_flux > 0
                    else 0
                )

                output.append(d)

            except Exception as e:

                print(
                    f"BLS Failed : {e}"
                )

        return output