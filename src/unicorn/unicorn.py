from multiprocessing import Queue
import numpy as np
import UnicornPy

from src.constants import SEGMENT_LEN_S


def connect_to_unicorn() -> UnicornPy.Unicorn:
    devices = UnicornPy.GetAvailableDevices(True)
    device = UnicornPy.Unicorn(devices[0])
    return device


def acquire_eeg_data_record(device, n_scans: int, buffer: bytearray, buffer_len: int,
                            n_channels: int) -> np.ndarray:
    device.GetData(n_scans, buffer, buffer_len)
    data = np.frombuffer(buffer, dtype=np.float32, count=n_channels * n_scans)
    return data


def acquire_eeg(queue: Queue, img_queue: Queue, fs: int, scan_len_s: int = 1) -> None:

    print(f'Connecting to Unicorn...')
    device = connect_to_unicorn()
    n_scans = scan_len_s * fs
    n_channels = device.GetNumberOfAcquiredChannels()
    buffer_len_bytes = n_scans * n_channels * 4
    print(f'Buffer length will be: {buffer_len_bytes} bytes')
    buffer = bytearray(buffer_len_bytes)

    device.StartAcquisition(False)

    while True:
        try:
            full_buffer = []
            for segment_nr in range(SEGMENT_LEN_S):
                data = acquire_eeg_data_record(device, n_scans=n_scans, buffer=buffer,
                                               buffer_len=buffer_len_bytes, n_channels=n_channels)
                data = np.reshape(data, (n_scans, n_channels))
                img_queue.put(data)
                full_buffer.append(data)
            queue.put(np.concatenate(full_buffer))
            print(f'Collected data to queue.')
        except Exception as e:
            print(e)
            device.StopAcquisition()
            raise


if __name__ == "__main__":
    from queue import Queue as FakeQueue
    q, q2 = FakeQueue(), FakeQueue()
    acquire_eeg(q, q2, fs=250)
