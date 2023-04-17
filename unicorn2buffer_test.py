import time
import multiprocessing

from loops import acquire_eeg


def print_got_data(eeg_queue: multiprocessing.Queue):
    while True:
        if not eeg_queue.empty():
            print(f'{eeg_queue.qsize()} records accumulated')
        print('Waiting for the recording to start...')
        time.sleep(0.5)


if __name__ == '__main__':

    RECORD_SIZE_S = 1

    eeg_queue = multiprocessing.Queue()
    acquisition_process = multiprocessing.Process(target=acquire_eeg, args=(eeg_queue, RECORD_SIZE_S))
    printing_process = multiprocessing.Process(target=print_got_data, args=(eeg_queue,))

    acquisition_process.start()
    printing_process.start()
