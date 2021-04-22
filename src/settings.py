import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "external", "processed")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")

META_DATA_LOC = os.path.join(INTERMEDIATE_DIR, "meta_data.csv")
MODEL_LOC = os.path.join(INTERMEDIATE_DIR, "my_dummy_model")
NORMALIZER_LOC = os.path.join(INTERMEDIATE_DIR, "normalizer.pickle")

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
