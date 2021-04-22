#!/usr/bin/env python
# coding: utf-8

# ## Start Feature Extraction from the collected Dataset

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from plot_helper import PlotHelp
from real_time_inference import RecordThread

# ## Model Building
from src.model import Net
from src.settings import DATA_DIR, PROCESSED_DIR
from src.utils import norm_spec

# os.chdir("../")


file_path = PROCESSED_DIR
meta_data = pd.read_csv(os.path.join(DATA_DIR, "meta_data.csv"))


## if this doesn't hold .. some deep problem we gotta fix my myan
assert all(meta_data.start_time.isna() == meta_data.end_time.isna())


meta_data["label"] = ~meta_data.start_time.isna()
meta_data["label"] = meta_data["label"].astype(int)


def get_melspectrogram_db(
    file_path,
    sr=44100,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8300,
    top_db=80,
):
    wav, sr = librosa.load(file_path, sr=sr)
    spec = librosa.feature.melspectrogram(
        wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


plot_help = PlotHelp()


# ## Check the Traning Example Visually


train_path = os.path.join(PROCESSED_DIR, "train")
test_path = os.path.join(PROCESSED_DIR, "test")

meta_data_train = meta_data[:1701]
meta_data_test = meta_data[1701:]

pos_examples = meta_data_train[meta_data_train.label == 1].sample(4)
neg_examples = meta_data_train[meta_data_train.label == 0].sample(4)

mel_spec_pos = [
    get_melspectrogram_db(os.path.join(train_path, pos_example.filename), 44100)
    for _, pos_example in pos_examples.iterrows()
]
mel_spec_pos_norm = [norm_spec(i) for i in mel_spec_pos]

mel_spec_neg = [
    get_melspectrogram_db(os.path.join(train_path, neg_example.filename), 44100)
    for _, neg_example in neg_examples.iterrows()
]
mel_spec_neg_norm = [norm_spec(i) for i in mel_spec_neg]

mel_scale_max = mel_spec_pos[0].shape[1]
time_scale_max = 8000  ## ms

potential_range_in_freq_domain = [
    (
        mel_scale_max * pos_example.start_time / time_scale_max,
        mel_scale_max * pos_example.end_time / time_scale_max,
    )
    for _, pos_example in pos_examples.iterrows()
]


plot_help.plot_examples(mel_spec_pos_norm, potential_range_in_freq_domain)


meta_data[:1701].shape, meta_data[1701:].shape


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


# Transformation using Librosa
audio_transformation = transforms.Compose(
    [
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        # lambda x: norm_spec(x),
        lambda x: x.reshape(1, 128, 690)
        # lambda x: Tensor(x)
    ]
)

# Transformation in Training Set
from audio_transformations import waveform_augment

training_transformation = transforms.Compose(
    [
        lambda x: waveform_augment(x, 44100),
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        # lambda x: norm_spec(x),
        lambda x: x.reshape(1, 128, 690)
        # lambda x: Tensor(x)
    ]
)


BATCH_SIZE = 8
# todo: multiprocessing, padding data
trainloader = DataLoader(
    AudioLoader(
        meta_data=meta_data_train, transform=training_transformation, mode="train"
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# todo: multiprocessing, padding data
testloader = DataLoader(
    AudioLoader(meta_data=meta_data_test, transform=audio_transformation, mode="test"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device to train : ", device)


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

    print(f"Saving at Epoch  {epoch} ..")
    torch.save(model.state_dict(), "my_dummy_model")
