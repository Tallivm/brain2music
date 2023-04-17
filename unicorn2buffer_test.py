import time

from multiprocessing import Queue, Process
from unicorn_utils import connect_to_unicorn, start_eeg_acquisition, stop_eeg_acquisition


def print_got_data(eeg_queue):
    while True:
        if eeg_queue:
            print(f'{len(eeg_queue)} records accumulated')
        time.sleep(0.01)


if __name__ == '__main__':
    device = connect_to_unicorn()

    eeg_queue = Queue()
    n_scans = 1

    try:
        acquisition_process = Process(target=start_eeg_acquisition, args=(device, eeg_queue, n_scans))
        printing_process = Process(target=print_got_data, args=(eeg_queue,))

        acquisition_process.start()
        printing_process.start()
    finally:
        stop_eeg_acquisition(device)
