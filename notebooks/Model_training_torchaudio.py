#!/usr/bin/env python
# coding: utf-8

# ## Start Feature Extraction from the collected Dataset

# In[1]:


import os
import pandas as pd
import librosa
import librosa.display ## To deal with module `display` not found error

import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
os.chdir('../')


# In[2]:


## This is necessary to find `src` folder
import sys
sys.path.append('C:\\Users\\Robin\\Downloads\\Podcast-Audio-Processing')


# In[3]:


import torch
torch.cuda.is_available()


# In[4]:


from plot_helper import PlotHelp
from src.real_time_inference import RecordThread


# In[5]:


from src.settings import META_DATA_LOC


# In[6]:


meta_data = pd.read_csv(META_DATA_LOC)


# In[7]:


## if this doesn't hold .. some deep problem we gotta fix my myan
assert all(meta_data.start_time.isna() == meta_data.end_time.isna())


# In[8]:


meta_data['label'] = ~meta_data.start_time.isna()
meta_data['label'] = meta_data['label'].astype(int) 


# In[9]:


import torchaudio


# In[10]:


def get_melspectrogram_db(file_path, sr=8000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path,sr=sr)
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db


# In[ ]:


def get_melspectrogram_db_torch():
    torchaudio.load('OSR_us_000_0010_8k.wav')


# ## Enter Torch Audio

# In[20]:


from IPython.display import Audio, display


# In[21]:


test_tensor, test_sr = torchaudio.load('data/generated/0.wav')


# In[22]:


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        return Audio(waveform[0], rate=sample_rate)
    elif num_channels == 2:
        return Audio((waveform[0], waveform[1]), rate=sample_rate)
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


# In[34]:


## PLAY RAW AUDIO
play_audio( test_tensor, test_sr )


# ### Some Augmentations with torchaudio

# In[40]:


# import random

# class RandomSpeedChange:
#     def __init__(self, sample_rate):
#         self.sample_rate = sample_rate

#     def __call__(self, audio_data):
#         speed_factor = random.choice([3, 4, 5])
#         if speed_factor == 1.0: # no change
#             return audio_data

#         # change speed and resample to original rate:
#         sox_effects = [
#             ["speed", str(speed_factor)],
#             ["rate", str(self.sample_rate)],
#         ]
#         transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
#             audio_data, self.sample_rate, sox_effects)
#         return transformed_audio

     
# speed_transformer = RandomSpeedChange(test_sr)
# transformed_audio = speed_transformer(test_tensor)


# In[41]:


# plot_help = PlotHelp()


# ## Check the Traning Example Visually  

# In[42]:


label_mapper = {
        'None' : 0,
        'start' : 1,
        'pause' : 2,
    }

meta_data['label'] = meta_data.label_type.map(label_mapper) 
meta_data.to_csv(META_DATA_LOC,index=False)


# In[43]:


meta_data.sample(7)


# ## Plot the Spectrograms

# In[44]:


## by default sr value is : 22500
sample, _ = librosa.load('data/generated/1.wav')
sample_duration = librosa.get_duration( sample)

train_path = 'data/generated/'

## This will have a huge effect, 
display_sr = 16000


# In[45]:


meta_data_train = meta_data[:1701]
meta_data_test = meta_data[1701:]

pos_examples = meta_data_train[meta_data_train.label == 1].sample(4)
neg_examples = meta_data_train[meta_data_train.label == 0].sample(4)
pause_examples = meta_data_train[meta_data_train.label == 2].sample(4)

mel_spec_pos = [ get_melspectrogram_db( os.path.join( train_path , pos_example.filename ) , display_sr  )  for _,pos_example in pos_examples.iterrows() ]

mel_spec_neg = [ get_melspectrogram_db( os.path.join( train_path , neg_example.filename ) , display_sr  ) for _,neg_example in neg_examples.iterrows() ]

mel_spec_pause = [ get_melspectrogram_db( os.path.join( train_path , pause_example.filename ) , display_sr  ) for _,pause_example in pause_examples.iterrows() ]


mel_scale_max = mel_spec_pos[0].shape[1]

def plot_samples(sample_df, spectrograms):
    ################################
    ###### IMPORTANT VARIABLE ######
    ################################

    ## We want this in Millisecond, becase that's how `start_time` and `end_time` is represented in `meta_data.csv`
    time_scale_max = sample_duration * 1000 

    potential_range_in_freq_domain = [  ( mel_scale_max * example.start_time / time_scale_max , 
                                          mel_scale_max * example.end_time / time_scale_max) 
                                          for _, example in sample_df.iterrows()]


    ### PLOT USING LIBROSA
    fig, ax = plt.subplots( len(sample_df) , 1, figsize=(10,16))
    for r in sample_df.reset_index().iterrows():
        row_ind = r[0]
        ax[row_ind].axvline( potential_range_in_freq_domain[ row_ind ][0], linewidth=2, color='red' )
        ax[row_ind].axvline( potential_range_in_freq_domain[ row_ind ][1], linewidth=2, color='red' )
        
        
        title_name = str(round(r[1].start_time,2)) + '---' + str(round(r[1].end_time,2)) + '----' + r[1].label_type
        ax[row_ind].set_title( 
            title_name
            )

        librosa.display.specshow(spectrograms[row_ind], ax=ax[row_ind] )    


plot_samples( pos_examples , mel_spec_pos )
plot_samples( pause_examples , mel_spec_pause )
plot_samples( neg_examples , mel_spec_neg )


# In[46]:


print('INPUT SHAPE : ', mel_spec_neg[0].shape )


# In[47]:


from pydub.playback import play
from pydub import AudioSegment

## Check one such example for Verification
pos_sound = AudioSegment.from_wav(os.path.join( train_path , pos_examples.iloc[0].filename ))
neg_sound = AudioSegment.from_wav(os.path.join( train_path , neg_examples.iloc[0].filename ))

def check_example(sound):

    ## playing positive example
    print('Playing ...')
    play( sound )
    print('Playing Stopped')

check_example(neg_sound)


# ## Data Transformations

# In[48]:


from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from src.settings import PROCESSED_DIR


# In[97]:


BATCH_SIZE = 12
NUM_WORKERS = 2

SAMPLE_RATE = 16000


# In[98]:


class AudioLoader(Dataset):
    def __init__(self, transform=None, mode="train"):
        # setting directories for data
        self.mode = mode
        self.audio_data_dir = PROCESSED_DIR
        meta_data = pd.read_csv(META_DATA_LOC)
        if self.mode == "train":
            self.meta_data = meta_data[:2001]
        elif self.mode == "test":
            self.meta_data = meta_data[2001:]
        else :
            self.meta_data = meta_data[:50]
        self.transform = transform

    def __len__(self):
        return self.meta_data.shape[0]

    def __getitem__(self, idx):
        filename = self.meta_data["filename"].iloc[idx]
        file_path = os.path.join(self.audio_data_dir, filename)

        data, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        data = self.transform(data)
        label = self.meta_data["label"].iloc[idx]

        return data, label


# training_transformation = transforms.Compose(
#     [
#         lambda x: waveform_augment(x, SAMPLE_RATE),
#         lambda x: librosa.feature.melspectrogram(
#             x, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
#         ),
#         lambda x: librosa.power_to_db(x, top_db=80),
# #         lambda x: (x - norm_dict["global_mean"]) / norm_dict["global_std"],
# #         lambda x: CustomFreqMask(fill_val=x.min())(x),
#         lambda x: x.reshape(1, x.shape[0], x.shape[1])
#     ]
# )


### CUSTOM AUGMENTATION ###
from audiomentations import (AddGaussianNoise, Compose, PitchShift, Shift,
                             SpecFrequencyMask, TimeMask, TimeStretch)

waveform_augment = Compose(
    [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
        # Fills to Minimum value of waveform
        TimeMask(min_band_part=0.1, max_band_part=0.15, fade=False, p=0.2),
        TimeStretch(min_rate=0.7, max_rate=1.25, p=0.3),
#         PitchShift(min_semitones=-4, max_semitones=4, p=1),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.3),
    ]
)


def training_transformation(x):
    x = waveform_augment(x, SAMPLE_RATE)
    x = librosa.feature.melspectrogram(
            x, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        )
    x = librosa.power_to_db(x, top_db=80)
    x = x.reshape(1, x.shape[0], x.shape[1])
    return x

trainloader = DataLoader(
    AudioLoader(transform=training_transformation, mode="train"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

# validation_transformation = transforms.Compose(
#     [
#         lambda x: librosa.feature.melspectrogram(
#             x, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
#         ),
#         lambda x: librosa.power_to_db(x, top_db=80),
# #         lambda x: (x - norm_dict["global_mean"]) / norm_dict["global_std"],
#         lambda x: x.reshape(1, x.shape[0], x.shape[1]),
#     ]
# )

def validation_transformation(x):
    x = librosa.feature.melspectrogram(
            x, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        )
    x = librosa.power_to_db(x, top_db=80)
    x = x.reshape(1, x.shape[0], x.shape[1])
    return x

# todo: multiprocessing, padding data
testloader = DataLoader(
    AudioLoader(transform=validation_transformation, mode="test"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    )

train_small_loader = DataLoader(
    AudioLoader(transform=training_transformation, mode="train_small"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    )


# ## Model Building

# In[101]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[102]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.drp = nn.Dropout2d(0.2)
        self.fc_drp = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool3 = nn.MaxPool2d(2, stride=3)

        self.conv1 = nn.Conv2d(
            1, 24, 5, dilation=1, stride=1
        )  # inchannel , outchannel , kernel size ..
        self.conv1_bn = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 32, 7, dilation=1, stride=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 9, dilation=1, stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 11, dilation=1, stride=2)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
#         print(x.shape)
        x = self.conv1_bn(self.pool1(F.relu(self.conv1(x))))
#         print(x.shape)
        x = self.drp(x)

        x = self.conv2_bn(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(x)

        x = self.conv3_bn(self.pool2(F.relu(self.conv3(x))))
        x = self.drp(x)

        x = self.conv4_bn(self.pool3(F.relu(self.conv4(x))))
        x = self.drp(x)

        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc_drp(x)

        x = self.fc2(x)

        return x


# In[103]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print( "USING : " , device)


# In[104]:


# defining the model
model = Net().to(device)
# defining the optimizer
optimizer = torch.optim.Adam(model.parameters())
# defining the loss function
criterion = nn.CrossEntropyLoss ().to(device)
# checking if GPU is available
print(model)


# ## Training the model

# ### First do a Overfitting Run
# 
# Decrease Trend of Loss in a small Sample ensures that Model indeed has a Capacity to Learn

# In[105]:


def calc_accuracy(outputs, labels):
    total_examples = len(outputs)
    correct_pred = torch.sum((outputs >= 0.5) == labels).to("cpu").item()
    return correct_pred / total_examples


# In[107]:

if __name__ == '__main__':


    for epoch in range(10):  # loop over the dataset multiple times

        model.train()
        running_loss = 0.0
        training_acc = []
        val_acc = []
        for i, data in enumerate(train_small_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
    #         labels = labels.unsqueeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            labels = labels.type(torch.long)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            training_acc.append( np.mean((torch.argmax(outputs,axis=1) == labels).cpu().numpy()) )

            if i %  10 == 0 :    #
                curr_training_loss = sum(training_acc)/len(training_acc)
                print( f'At {i+1}th Epoch, iter {epoch+1} :  Loss accumulated upto : {running_loss} || Running Train Accuracy : {curr_training_loss}' )
            


    # # ## DO REAL TRAINING HERE

    # # In[47]:


    # for epoch in range(1):  # loop over the dataset multiple times
        
    #     ### Training Part ###
    #     model.train()
    #     curr_training_loss = 0.0
    #     training_acc = []
        
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         labels = labels.type(torch.long)

    #         loss = criterion(outputs, labels)

    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         curr_training_loss += loss.item()
    #         training_acc.append( np.mean((torch.argmax(outputs,axis=1) == labels).cpu().numpy()) )

    #     ### Evaluation Part ###
    #     model.eval()
    #     curr_val_loss = 0.0
    #     val_acc = []
        
    #     for i, data in enumerate(testloader, 0):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         output_val = model(inputs)
    #         labels = labels.type(torch.long)
    #         loss_val = criterion(output_val, labels)

    #         curr_val_loss += loss_val.item()
    #         val_acc.append(  np.mean((torch.argmax(outputs,axis=1) == labels).cpu().numpy()) ) 

    #     combined_training_acc = sum(training_acc)/len(training_acc)
    #     combined_val_acc = sum(val_acc)/len(val_acc)

    #     print(f'After Epoch {i+1} : Training Loss {curr_training_loss} || Validation loss {curr_val_loss}')
    #     print(f'Training Accuracy {combined_training_acc} || Validation Accuracy {combined_val_acc}')

    # print('Finished Training . now saving ')
    # torch.save(model.state_dict(), 'my_dummy_model')

