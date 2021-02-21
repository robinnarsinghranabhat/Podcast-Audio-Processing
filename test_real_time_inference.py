from real_time_inference import RecordThread, TestThread
import time

record = RecordThread('sample_record.wav', 4)
print(record.start())

print('Recording in background, printing in foreground ..')
time.sleep(5)

record.stoprecord()
print('Recording Stopped')




# SAMPLE TEST CLASS USING THREADS ##

# test = TestThread(12, 24)
# test.start()

# print('out of thread ... computation')
# time.sleep(2)
# print('dummy task completed')

# print('stopping thread')
# test.terminate_thread()
