import glob

from dataset_pipeline import pipeline

from transformers.feature_extractor import FeatureExtractor
from transformers.phase_folder import PhaseFolder

fits_files = glob.glob(
    "raw_tess/*.fits"
)

processed = pipeline.fit_transform(
    fits_files
)

features = FeatureExtractor(
).fit_transform(
    processed
)

features.to_csv(
    "features.csv",
    index=False
)

PhaseFolder(
    save_dir="curves",
    n_points=200
).fit_transform(
    processed
)

print(
    "features.csv saved"
)

print(
    "curves folder saved"
)