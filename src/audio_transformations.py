from audiomentations import (AddGaussianNoise, Compose, PitchShift, Shift,
                             SpecFrequencyMask, TimeMask, TimeStretch)

waveform_augment = Compose(
    [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
        # Fills to Minimum value of waveform
        TimeMask(min_band_part=0.1, max_band_part=0.15, fade=False, p=0.8),
        # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
        # PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.4),
    ]
)


def CustomFreqMask(fill_val=0):
    return SpecFrequencyMask(fill_constant=fill_val, p=0.8)
