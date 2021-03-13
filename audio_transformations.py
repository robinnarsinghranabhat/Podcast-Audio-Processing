import audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, FrequencyMask
import numpy as np

## Each of these possible Transformation has probability of 50 percent
## WaveForm Transform ##

waveform_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.4),
    ])
