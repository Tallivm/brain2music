import numpy as np

import UnicornPy


def connect_to_unicorn() -> UnicornPy.Unicorn:
    devices = UnicornPy.GetAvailableDevices(True)
    device = UnicornPy.Unicorn(devices[0])
    return device


def calculate_buffer_len(n_scans: int, n_channels: int, buffer_size: int = 10) -> int:
    receive_buffer_length = n_scans * n_channels * buffer_size
    return receive_buffer_length


def buffer2numpy(buffer: bytearray, n_scans: int, n_channels: int) -> np.ndarray:
    return np.frombuffer(buffer, dtype=np.float32, count=n_channels * n_scans)


def acquire_eeg_data_record(device, n_scans: int, buffer: bytearray, buffer_len: int,
                            n_channels: int) -> np.ndarray:
    device.GetData(n_scans, buffer, buffer_len)
    data = buffer2numpy(buffer, n_scans, n_channels)
    return data


def stop_eeg_acquisition(device: UnicornPy.Unicorn) -> None:
    device.StopAcquisition()


def reshape_eeg_record(data: np.ndarray, n_scans: int, n_channels: int) -> np.ndarray:
    return np.reshape(data, (n_scans, n_channels))
