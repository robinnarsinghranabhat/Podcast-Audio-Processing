from real_time_inference import RecordThread, TestThread
import time

# record = RecordThread('sample_record.wav')

# print(record.run())

# print('Recording in vacxkground, printing in foreground ..')
# time.sleep(7)

# record.stoprecord()
# print('Recording Stopped')

test = TestThread()
test.start(3)

print('out of thread ... computation')
time.sleep(2)
print('rast comp[leted')

print('stopping thread')
test.terminate_thread()