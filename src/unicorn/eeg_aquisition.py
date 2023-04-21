from multiprocessing import Queue


def acquire_eeg(queue: Queue, record_size: int, fs: int) -> None:
    from src.unicorn.unicorn_utils import connect_to_unicorn, calculate_buffer_len, acquire_eeg_data_record, \
        reshape_eeg_record
    print(f'Connecting to Unicorn...')
    device = connect_to_unicorn()
    n_scans = record_size * fs
    n_channels = device.GetNumberOfAcquiredChannels()
    buffer_len = calculate_buffer_len(n_scans, n_channels, buffer_size=4)
    print(f'Buffer length will be: {buffer_len}')
    buffer = bytearray(buffer_len)

    device.StartAcquisition(False)

    while True:
        try:
            data = acquire_eeg_data_record(device, n_scans, buffer, buffer_len, n_channels)
            data = reshape_eeg_record(data, n_scans, n_channels)
            queue.put(data)
            print('Collected data sample')
        except Exception as e:
            print(e)
            device.StopAcquisition()
            break
