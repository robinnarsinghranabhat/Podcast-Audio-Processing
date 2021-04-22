import pickle

import librosa
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.audio_loaders import AudioLoader
from src.settings import NORMALIZER_LOC

normalization_transformation = transforms.Compose(
    [
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: x.reshape(1, x.shape[0], x.shape[1]),
    ]
)

normalization_loader = DataLoader(
    AudioLoader(transform=normalization_transformation, mode="train"),
    batch_size=32,
    shuffle=True,
    num_workers=0,
)


def calculate_norm_params():
    # Get GLobal Normalization Params :
    print("Calculating Global MEAN and STD .. takes a while..")
    mean_spec = []
    std_spec = []
    for i, data in enumerate(normalization_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data[0], data[1]
        batch_mean = torch.mean(inputs)
        batch_std = torch.std(inputs)
        mean_spec.append(batch_mean.numpy())
        std_spec.append(batch_std.numpy())

    global_mean = sum(mean_spec) / len(mean_spec)
    global_std = sum(std_spec) / len(std_spec)

    global_normalization_dict = {"global_mean": global_mean, "global_std": global_std}

    print(global_normalization_dict)
    print("Saving global normalization paramaters for inference .. ")
    with open(NORMALIZER_LOC, "wb") as handle:
        pickle.dump(global_normalization_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
