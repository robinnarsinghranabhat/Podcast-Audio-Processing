# coding=utf-8
# Python3.6
# Class Record a wav in new thread
# Author:Why
# Date:2018.04.23

import math
import threading
import time
import wave

import pyaudio


class RecordThread(threading.Thread):
    def __init__(self, *args):

        threading.Thread.__init__(self)

        self.audiofile, self.recording_interval = args
        self.bRecord = True
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100

    # Defining a Class Variable
    save_name_counter = 0

    def run(self):
        audio = pyaudio.PyAudio()
        wavfile = wave.open(self.audiofile, "wb")
        wavfile.setnchannels(self.channels)
        wavfile.setsampwidth(audio.get_sample_size(self.format))
        wavfile.setframerate(self.rate)
        wavstream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        if self.recording_interval is None:
            print("Recording silently .. :)")
            while self.bRecord:
                wavfile.writeframes(wavstream.read(self.chunk))
            # noqa
            wavstream.stop_stream()
            wavstream.close()
            audio.terminate()
        else:
            print(
                f"Recording and Saving at {self.recording_interval} seconds Gap"
            )  ## noqa
            while self.bRecord:
                frames = []  # Initialize array to store frames

                # Soundcard mic intercepts audio as chunk of 1024 sample
                # We need to collect total of recording_time * 44100 samples
                # So for a recording_time seconds of audio, mic needs to
                # collect chunks: (recording_time * 44100) / 1024
                max_range_interval = math.ceil(
                    self.rate / self.chunk * self.recording_interval
                )
                for i in range(0, max_range_interval):
                    if i == max_range_interval - 1:
                        collected = 1024 * (max_range_interval - 1)
                        needed = self.rate * self.recording_interval
                        extra_chunks = needed - collected
                        assert extra_chunks > 0
                        data = wavstream.read(extra_chunks)
                        frames.append(data)
                        continue
                    data = wavstream.read(self.chunk)
                    frames.append(data)

                print("Finished recording an interval ")
                print("Saving the interval")
                save_name = f"inference_{RecordThread.save_name_counter}.wav"
                save_thread = SaveRecordingThread(
                    self.channels,
                    audio.get_sample_size(self.format),
                    self.rate,
                    save_name,
                    frames,
                )  # noqa
                save_thread.start()

    def stoprecord(self):
        self.bRecord = False


class SaveRecordingThread(threading.Thread):
    def __init__(self, *args):
        threading.Thread.__init__(self)
        (
            self.channels,
            self.sample_width,
            self.rate,
            self.filename,
            self.frames_to_write,
        ) = args  # noqa

    def run(self):

        # Save the recorded data as a WAV file
        wf = wave.open(self.filename, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.sample_width)
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(self.frames_to_write))
        wf.close()

        # Update Class Variable
        # Note , Class Varaible can only be updated by Class name
        # RecordThread.save_name_counter += 1


class TestThread(threading.Thread):
    def __init__(self, *args):
        threading.Thread.__init__(self)
        self.thread_switch = True
        self.outer_args_1, self.outer_args_2 = args

    def run(self):
        while self.thread_switch:
            print(f"Backround running . Got : {self.outer_args_1, self.outer_args_2}")
            # time.sleep(4)

    def terminate_thread(self):
        self.thread_switch = False
