from sklearn.base import BaseEstimator, TransformerMixin
from astropy.io import fits
import numpy as np

class FluxExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, fits_files):

        output = []

        for file in fits_files:

            try:

                with fits.open(file) as hdul:

                    lc = hdul[1].data

                    time = lc["TIME"]
                    flux = lc["PDCSAP_FLUX"]
                    quality = lc["QUALITY"]

                    mask = (
                        np.isfinite(time)
                        & np.isfinite(flux)
                    )

                    output.append({

                        "file": file,

                        "time": time[mask],

                        "flux": flux[mask],

                        "quality": quality[mask]

                    })

            except Exception as e:

                print(file, e)

        return output