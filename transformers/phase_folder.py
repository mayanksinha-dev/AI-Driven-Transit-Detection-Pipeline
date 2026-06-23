from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os

class PhaseFolder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        save_dir="curves",
        n_points=200
    ):

        self.save_dir = save_dir
        self.n_points = n_points

        os.makedirs(
            save_dir,
            exist_ok=True
        )

    def fit(self, X, y=None):
        return self

    def transform(self, data):

        for d in data:

            try:

                phase = (
                    d["time"] %
                    d["period"]
                ) / d["period"]

                idx = np.argsort(
                    phase
                )

                phase = phase[idx]

                flux = d["flux"][idx]

                phase_grid = np.linspace(
                    0,
                    1,
                    self.n_points
                )

                folded_flux = np.interp(
                    phase_grid,
                    phase,
                    flux
                )

                folded_flux = (
                    folded_flux -
                    np.mean(
                        folded_flux
                    )
                ) / (
                    np.std(
                        folded_flux
                    ) + 1e-8
                )

                tic = os.path.basename(
                    d["file"]
                ).replace(
                    ".fits",
                    ""
                )

                np.savez_compressed(

                    os.path.join(
                        self.save_dir,
                        f"{tic}.npz"
                    ),

                    phase=phase_grid,

                    flux=folded_flux,

                    period=d["period"],

                    duration=d["duration"],

                    depth=d["depth"],

                    transit_time=d["transit_time"],

                    snr=d["transit_snr"],

                    power=d["bls_power"]

                )

            except Exception as e:

                print(
                    f"Curve Save Failed : {e}"
                )

        return data