import os
import pandas as pd
import pickle
import librosa

import torchaudio
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from audio_transformations import waveform_augment
from src.settings import PROCESSED_DIR, DATA_DIR

BATCH_SIZE = 20


class AudioLoader(Dataset):
    def __init__(self, transform=None, mode='train'):
        # setting directories for data
        self.mode = mode
        self.audio_data_dir = os.path.join(PROCESSED_DIR, self.mode) 
        self.meta_data = pd.read_csv(os.path.join(DATA_DIR, 'meta_data.csv'))
        if self.mode == 'train':
            self.meta_data = self.meta_data[:2001]
        else:
            self.meta_data = self.meta_data[2001:]
        self.transform = transform

    def __len__(self):
        return self.meta_data.shape[0]

    def __getitem__(self, idx):
        filename = self.meta_data["filename"].iloc[idx]
        file_path = os.path.join(self.audio_data_dir, filename)

        data, sr = librosa.load(file_path, sr=44100)

        if self.transform is not None:
            data = self.transform(data)

        label = self.meta_data["label_type"].iloc[idx]

        if label == 'activate':
            label = 2
        elif label == 'start':
            label = 1
        else:
            label = 0

        return data, label


# Define Different Transformations
if 'normalizer.pickle' in os.listdir('.'):
    print('Opeining normalizer')
    with open('normalizer.pickle', 'rb') as handle:
        norm_dict = pickle.load(handle)
else:
    from get_normalization_params import calculate_norm_params
    calculate_norm_params()
    with open('normalizer.pickle', 'rb') as handle:
        norm_dict = pickle.load(handle)

training_transformation = transforms.Compose(
    [
        lambda x: waveform_augment(x, 44100),
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: (x - norm_dict['global_mean']) / norm_dict['global_std'],
        lambda x: x.reshape(1, x.shape[0], x.shape[1])
        # lambda x: Tensor(x)
    ]
)

# todo: multiprocessing, padding data
trainloader = DataLoader(
    AudioLoader(
        transform=training_transformation, mode="train"
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_transformation = transforms.Compose(
    [   
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: (x - norm_dict['global_mean']) / norm_dict['global_std'],
        lambda x: x.reshape(1, x.shape[0], x.shape[1])
    ]
)

# todo: multiprocessing, padding data
testloader = DataLoader(
    AudioLoader(transform=validation_transformation, mode="test"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

