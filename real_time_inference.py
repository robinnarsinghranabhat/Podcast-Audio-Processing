# coding=utf-8
# Python3.6
# Class Record a wav in new thread
# Author:Why
# Date:2018.04.23

import threading
import pyaudio
import wave
import time


class RecordThread(threading.Thread):

    def __init__(self, *args):

        threading.Thread.__init__(self)

        self.bRecord = True
        self.audiofile = args[0]
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.recording_interval = args[1]

        self.audiofile,, self.sample_width,  self.filename, self.frames_to_write 

    # Defining a Class Variable
    save_name_counter = 0

    def run(self):
        audio = pyaudio.PyAudio()
        wavfile = wave.open(self.audiofile, 'wb')
        wavfile.setnchannels(self.channels)
        wavfile.setsampwidth(audio.get_sample_size(self.format))
        wavfile.setframerate(self.rate)
        wavstream = audio.open(format=self.format,
                               channels=self.channels,
                               rate=self.rate,
                               input=True,
                               frames_per_buffer=self.chunk)

        
        save_thread = SaveRecordingThread( (self.channels, audio.get_sample_size(self.format), self.rate) ) # noqa

        if recording_interval is None:
            print('Recording silently .. :)')              
            while self.bRecord:
                wavfile.writeframes(wavstream.read(self.chunk))
            # noqa
            wavstream.stop_stream()
            wavstream.close()
            audio.terminate()
        else:
            print(f'Recording Silently for {recording_interval} seconds')
            frames = []  # Initialize array to store frames

            # Soundcard mic intercepts audio as chunk of 1024 sample
            # We need to collect total of recording_time * 44100 samples
            # So for a recording_time seconds of audio, mic needs to
            # collect chunks: (recording_time * 44100) / 1024
            max_range_interval = int(self.rate / self.chunk * recording_interval)
            for i in range(0, max_range_interval):
                data = wavstream.read(self.chunk)
                frames.append(data)

            print('Finished recording an interval ')
            print('Saving the interval')

            save_name = f'inference_{RecordThread.save_name_counter}.wav'

            # wf = wave.open(filename, 'wb')
            # wf.setnchannels(self.channels)
            # wf.setsampwidth(audio.get_sample_size(self.format))
            # wf.setframerate( self.rate )
            # wf.writeframes(b''.join(frames))
            # wf.close()
            save_thread.run(save_name, frames)

            # Update Class Variable
            # Note , Class Varaible can only be updated by Class name

            RecordThread.save_name_counter += 1

    def stoprecord(self):
        self.bRecord = False     


class SaveRecordingThread(threading.Thread):
    def __init__(self, *args):
        threading.Thread.__init__(self)
 
        self.channels, self.sample_width, self.rate, self.filename, self.frames_to_write = args # noqa

    def run(self):

        # Save the recorded data as a WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.sample_width)
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames_to_write))
        wf.close()


class TestThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, *args)
        self.thread_switch = True
        self.outer_args = args

    def run(self):
        while self.thread_switch:
            print(f'Backround running . Got : {self.outer_args}')
            time.sleep(4)
                                                                                   
    def terminate_thread(self):
        self.thread_switch = False
