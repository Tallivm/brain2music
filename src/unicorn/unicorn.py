from multiprocessing import Queue
import numpy as np
import UnicornPy


def connect_to_unicorn() -> UnicornPy.Unicorn:
    devices = UnicornPy.GetAvailableDevices(True)
    device = UnicornPy.Unicorn(devices[0])
    return device


def acquire_eeg_data_record(device, n_scans: int, buffer: bytearray, buffer_len: int,
                            n_channels: int) -> np.ndarray:
    device.GetData(n_scans, buffer, buffer_len)
    data = np.frombuffer(buffer, dtype=np.float32, count=n_channels * n_scans)
    return data


def acquire_eeg(queue: Queue, record_size: int, fs: int) -> None:

    print(f'Connecting to Unicorn...')
    device = connect_to_unicorn()
    n_scans = record_size * fs
    n_channels = device.GetNumberOfAcquiredChannels()
    buffer_len = n_scans * n_channels * 10
    print(f'Buffer length will be: {buffer_len}')
    buffer = bytearray(buffer_len)

    device.StartAcquisition(False)

    while True:
        try:
            data = acquire_eeg_data_record(device, n_scans, buffer, buffer_len, n_channels)
            data = np.reshape(data, (n_scans, n_channels))
            queue.put(data)
            print('Collected data sample')
        except Exception as e:
            print(e)
            device.StopAcquisition()
            break
