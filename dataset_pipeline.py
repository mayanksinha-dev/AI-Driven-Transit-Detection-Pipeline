from sklearn.pipeline import Pipeline

from transformers.flux_extractor import FluxExtractor
from transformers.quality_filter import QualityFilter
from transformers.mad_filter import MADFilter
from transformers.savgol_detrender import SavGolDetrender
from transformers.bls_search import BLSSearch

pipeline = Pipeline([

    ("flux", FluxExtractor()),

    ("quality", QualityFilter()),

    ("mad", MADFilter()),

    ("savgol", SavGolDetrender()),

    ("bls", BLSSearch())

])