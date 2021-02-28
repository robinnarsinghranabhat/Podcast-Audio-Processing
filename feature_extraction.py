#!/usr/bin/env python
# coding: utf-8

# ## Start Feature Extraction from the collected Dataset

from src.settings import DATA_DIR, PROCESSED_DIR
import os
import pandas as pd
import librosa

import numpy as np
import matplotlib.pyplot as plt

os.chdir("../")


import torch

torch.cuda.is_available()


from plot_helper import PlotHelp
from real_time_inference import RecordThread


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


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def norm_spec(spec):
    return (spec - spec.min()) / (spec.max() - spec.min())


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


mel_spec_neg_norm[0].shape


from pydub.playback import play
from pydub import AudioSegment

## Check one such example for Verification
pos_sound = AudioSegment.from_wav(
    os.path.join(train_path, pos_examples.iloc[2].filename)
)
neg_sound = AudioSegment.from_wav(
    os.path.join(train_path, neg_examples.iloc[0].filename)
)


def check_example(sound):

    ## playing positive example
    print("Playing ...")
    play(sound)
    print("Playing Stopped")


check_example(pos_sound)


# ## Sanity Check : Take in an audio input and take a look at spectrogram


# ## my personal recording
# import time
# record = RecordThread('sample_record.wav', 8)
# record.start()
# time.sleep(0.2)
# record.stoprecord()

# print(get_melspectrogram_db( 'inference_0.wav' , 44100  ).shape)

# plot_help.plot_examples( [norm_spec(get_melspectrogram_db( 'inference_0.wav' , 44100  ))] )


# ## Working with Audio Dataloaders and Transformations


from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


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


audio_transformation = transforms.Compose(
    [
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: norm_spec(x),
        lambda x: x.reshape(1, 128, 690)
        # lambda x: Tensor(x)
    ]
)

# todo: multiprocessing, padding data
trainloader = DataLoader(
    AudioLoader(
        meta_data=meta_data_train, transform=audio_transformation, mode="train"
    ),
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

# todo: multiprocessing, padding data
testloader = DataLoader(
    AudioLoader(meta_data=meta_data_test, transform=audio_transformation, mode="test"),
    batch_size=32,
    shuffle=True,
    num_workers=0,
)


# ## Model Building


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)  ## inchannel , outchannel , kernel size ..
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv1_bn = nn.BatchNorm2d(8)

        self.drp = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool2 = nn.MaxPool2d(4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 12 * 82, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1_bn(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(x)

        x = self.conv2_bn(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(x)

        x = self.conv3_bn(self.pool2(F.relu(self.conv3(x))))
        x = self.drp(x)
        # x = self.drp(self.pool1(F.relu(self.conv4(x))))
        # x = self.drp(self.pool2(F.relu(self.conv5(x))))
        # size = torch.flatten(x).shape[0]
        x = x.view(-1, 32 * 12 * 82)
        # x = x.unsqueeze_(1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# device = 'cpu'


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

        if i % 10 == 0:  #
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

print("Finished Training . now saving ")
torch.save(model.state_dict(), "my_dummy_model")