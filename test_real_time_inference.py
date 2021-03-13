from real_time_inference import RecordThread, TestThread
import time


from src.model import Net
import torch
device = 'cpu'
model = Net()
model.load_state_dict(torch.load('data\my_dummy_model'))
model.eval()


# THIS PUTS LATEST AUDIO COPY IN THE BACKGROUND
record = RecordThread('sample_record.wav', 8)
print(record.start())

print('Recording in background, infering in foreground ..')
# # time.sleep(20)


## load the latest global mean and variance
import pickle
with open('normalizer.pickle', 'rb') as handle:
    norm_dict = pickle.load(handle)



## load the latest chunk of audio input :
import librosa
from torchvision import transforms
from torch import Tensor
audio_transformation = transforms.Compose(
    [   
        lambda x: librosa.feature.melspectrogram(
            x, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300
        ),  # MFCC
        lambda x: librosa.power_to_db(x, top_db=80),
        lambda x: (x - norm_dict['global_mean']) / norm_dict['global_std'],
        lambda x: x.reshape(1, 128, 690)
    ]
)

import time

prev_data = None

while True:
    try:
        data, sr = librosa.load('inference_0.wav', sr=44100)
        if prev_data is not None and all(prev_data == data):
            continue
        prev_data = data.copy()

        data = audio_transformation(data)
        data = Tensor(data.reshape(-1, 1, 128, 690))
        out = model(data).item()
        # print(out)

        if out > 0.75: 
            print('Video Paused')

    except:
        # print('EXCEPTION GRACEFULLY HANDLED')
        continue
        
        
        

    

# record.stoprecord()
# print('Recording Stopped')
