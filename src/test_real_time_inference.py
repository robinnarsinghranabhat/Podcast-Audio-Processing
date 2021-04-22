import time

import torch

from real_time_inference import RecordThread, TestThread
from src.model import Net
from src.settings import MODEL_LOC

device = "cpu"
model = Net()
model.load_state_dict(torch.load(MODEL_LOC))
model.eval()


# THIS PUTS LATEST AUDIO COPY IN THE BACKGROUND
record = RecordThread("sample_record.wav", 4)
print(record.start())

print("Recording in background, infering in foreground ..")
# # time.sleep(20)


## load the latest global mean and variance
import pickle

with open("normalizer.pickle", "rb") as handle:
    norm_dict = pickle.load(handle)


## load the latest chunk of audio input :
import librosa
from torch import Tensor
from torchvision import transforms

audio_transformation = transforms.Compose(
    [
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: (x - norm_dict["global_mean"]) / norm_dict["global_std"],
        lambda x: x.reshape(1, x.shape[0], x.shape[1]),
    ]
)

import time

prev_data = None

while True:
    try:
        data, sr = librosa.load("inference_0.wav", sr=44100)
        # if prev_data is not None and all(prev_data == data):
        #     continue
        # prev_data = data.copy()

        data = audio_transformation(data)
        data = Tensor(data.reshape(-1, 1, 128, 345))

        out = model(data)
        out = torch.nn.functional.softmax(out)
        out_ind = torch.argmax(out).item()
        out_val = torch.max(out).item()
        # print(out)
        # import pdb; pdb.set_trace()
        if out_ind == 1 and out_val > 0.9:
            print("Started")
            time.sleep(0.5)

        if out_ind == 2 and out_val > 0.9:
            print("ACtivated")
            time.sleep(0.5)

        print(out_ind, out_val)
        time.sleep(2)

    except:
        continue


# record.stoprecord()
# print('Recording Stopped')
