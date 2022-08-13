import os, sys
from pathlib import Path, WindowsPath

if sys.platform == 'win32':
    ROOT_DIR = WindowsPath(__file__).parent.parent
else:
    ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "generated"
META_DATA_DIR = DATA_DIR / "meta_data_dir"
MODEL_DATA_LOC = DATA_DIR / "saved_models"  

META_DATA_LOC = META_DATA_DIR / "meta_data.csv"


os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(META_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DATA_LOC, exist_ok=True)

## OTHER CONFIGS
overfitting_check = False