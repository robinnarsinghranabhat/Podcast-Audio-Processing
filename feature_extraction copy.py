#!/usr/bin/env python
# coding: utf-8

# ## Start Feature Extraction from the collected Dataset

from src.settings import DATA_DIR, PROCESSED_DIR
from src.utils import norm_spec
import os
import pandas as pd
import librosa

import numpy as np
import matplotlib.pyplot as plt

# os.chdir("../")

# ## Model Building
from src.model import Net
import torch
import torch.nn as nn

from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


file_path = PROCESSED_DIR
meta_data = pd.read_csv(os.path.join(DATA_DIR, "meta_data.csv"))


## if this doesn't hold .. some deep problem we gotta fix my myan
assert all(meta_data.start_time.isna() == meta_data.end_time.isna())


meta_data["label"] = ~meta_data.start_time.isna()
meta_data["label"] = meta_data["label"].astype(int)



train_path = os.path.join(PROCESSED_DIR, "train")
test_path = os.path.join(PROCESSED_DIR, "test")

meta_data_train = meta_data[:1701]
meta_data_test = meta_data[1701:]

# Making Data loader

class AudioLoader(Dataset):
    def __init__(self, meta_data, transform=None, mode="train"):
        # setting directories for data
        data_root = PROCESSED_DIR
        self.mode = mode
        if self.mode is "train":
            self.data_dir = os.path.join(data_root, "train")
            self.csv_file = meta_data_train

        elif self.mode is "test":
            self.data_dir = os.path.join(data_root, "test")
            self.csv_file = meta_data_test

        self.transform = transform

    def __len__(self):
        return self.csv_file.shape[0]

    def __getitem__(self, idx):
        filename = self.csv_file["filename"].iloc[idx]
        file_path = os.path.join(self.data_dir, filename)

        data, sr = librosa.load(file_path, sr=44100)

        if self.transform is not None:
            data = self.transform(data)

        label = self.csv_file["label"].iloc[idx]
        return data, label

# import torchaudio
# # Pytorch Transformation
# torch_audio_transformation = transforms.Compose(
#     lambda x: torchaudio.transforms.MelSpectrogram(
#                sample_rate=44100, n_mels=128, n_fft=2048, f_max=12000, hop_length=512),
#     lambda x : torchaudio.transforms.AmplitudeToDB(top_db=80)
# )


BATCH_SIZE = 10

##################### NORMALIZATION OF SPECTROGRAM ###########################
normalization_transformation = transforms.Compose(
    [
        # lambda x: waveform_augment(x, 44100),
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: x.reshape(1, 128, 690)
    ]
)

LIMIT_TRAIN = 100000

normalization_loader = DataLoader(
    AudioLoader(
        meta_data=meta_data_train[:LIMIT_TRAIN], transform=normalization_transformation, mode="train"
    ),
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

# Get GLobal Normalization Params :
print('Calculating Global MEAN and STD .. takes a while..')
mean_spec = []
std_spec = []
for i, data in enumerate(normalization_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0], data[1]
    batch_mean = torch.mean(inputs)
    batch_std = torch.std(inputs)
    mean_spec.append(batch_mean.numpy())
    std_spec.append(batch_std.numpy())

try:
    plt.plot(mean_spec, kind='hist')
    plt.show()
    plt.plot(std_spec, kind='hist')
    plt.show()
except:
    pass


global_mean = sum(mean_spec) / len(mean_spec)
global_std = sum(std_spec) / len(std_spec)

global_normalization_dict = {'global_mean': global_mean, 'global_std': global_std}

print(global_normalization_dict)
print('Saving global normalization paramaters for inference .. ')
import pickle
with open('normalizer.pickle', 'wb') as handle:
    pickle.dump(global_normalization_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
from audio_transformations import waveform_augment
training_transformation = transforms.Compose(
    [
        lambda x: waveform_augment(x, 44100),
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: (x - global_mean) / global_std,
        lambda x: x.reshape(1, 128, 690)
        # lambda x: Tensor(x)
    ]
)

# todo: multiprocessing, padding data
trainloader = DataLoader(
    AudioLoader(
        meta_data=meta_data_train[:LIMIT_TRAIN], transform=training_transformation, mode="train"
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

audio_transformation = transforms.Compose(
    [   
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: (x - global_mean) / global_std,
        lambda x: x.reshape(1, 128, 690)
    ]
)

# todo: multiprocessing, padding data
testloader = DataLoader(
    AudioLoader(meta_data=meta_data_test[:LIMIT_TRAIN], transform=audio_transformation, mode="test"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device to train : ', device)


# defining the model
model = Net().to(device)
# defining the optimizer
optimizer = torch.optim.Adam(model.parameters())
# defining the loss function
criterion = nn.BCELoss().to(device)
# checking if GPU is available
print(model)


# ## Training the model
def calc_accuracy(outputs, labels):
    total_examples = len(outputs)
    correct_pred = torch.sum((outputs >= 0.5) == labels).to("cpu").item()
    return correct_pred / total_examples


for epoch in range(50):  # loop over the dataset multiple times

    model.train()
    running_loss = 0.0
    training_acc = []
    val_acc = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.unsqueeze(1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        labels = labels.type_as(outputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        training_acc.append(calc_accuracy(outputs, labels))

        if i % 50 == 0:  #
            curr_training_loss = sum(training_acc) / len(training_acc)
            print(
                f"At {i+1}th iter, Epoch {epoch+1} :  Loss accumulated upto : {running_loss} || Running Train Accuracy : {curr_training_loss}"
            )

    model.eval()
    val_loss = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = labels.unsqueeze(1)
        output_val = model(inputs)

        labels = labels.type_as(outputs)
        loss_val = criterion(output_val, labels)

        val_loss += loss_val.item()
        val_acc.append(calc_accuracy(output_val, labels))

    curr_training_loss = sum(training_acc) / len(training_acc)
    curr_val_loss = sum(val_acc) / len(val_acc)

    print(
        f"After Epoch {i+1} : Training Loss {running_loss} || Validation loss {val_loss}"
    )
    print(
        f"Training Accuracy {curr_training_loss} || Validation Accuracy {curr_val_loss}"
    )

    print(f"SAving at epoch {epoch} ")
    torch.save(model.state_dict(), "my_dummy_model")
