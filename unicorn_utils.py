from multiprocessing import Queue
import numpy as np

import UnicornPy


def connect_to_unicorn() -> UnicornPy.Unicorn:
    devices = UnicornPy.GetAvailableDevices(True)
    device = UnicornPy.Unicorn(devices[0])
    return device


def calculate_buffer_len(device: UnicornPy.Unicorn, n_scans: int) -> bytearray:
    num_channels = device.GetNumberOfAcquiredChannels()
    receive_buffer_length = n_scans * num_channels * 4
    return receive_buffer_length


def buffer2numpy(buffer: bytearray, n_scans: int, n_channels: int) -> np.ndarray:
    return np.frombuffer(buffer, dtype=np.float32, count=n_channels * n_scans)


def start_eeg_acquisition(device: UnicornPy.Unicorn, queue: Queue, n_scans: int) -> None:
    n_channels = device.GetNumberOfAcquiredChannels()
    buffer_len = calculate_buffer_len(device, n_scans)
    buffer = bytearray(buffer_len)

    device.StartAcquisition(bool_testSignalEnabledFlag=False)

    while True:
        device.GetData(n_scans, buffer, buffer_len)
        data = buffer2numpy(buffer, n_scans, n_channels)
        queue.put(data)


def stop_eeg_acquisition(device: UnicornPy.Unicorn) -> None:
    device.StopAcquisition()


def reshape_eeg_record(data: np.ndarray, n_scans: int, n_channels: int) -> np.ndarray:
    return np.reshape(data, (n_scans, n_channels))
