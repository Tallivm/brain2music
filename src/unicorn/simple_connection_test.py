import UnicornPy

from src.unicorn import unicorn_utils as uu


def test_eeg_acquisition(device: UnicornPy.Unicorn, n_scans: int, n_data_calls: int) -> None:
    n_channels = device.GetNumberOfAcquiredChannels()
    buffer_len = uu.calculate_buffer_len(n_scans, n_channels, buffer_size=4)
    print(f'Buffer length will be: {buffer_len}')
    buffer = bytearray(buffer_len)

    device.StartAcquisition(False)

    for i in range(n_data_calls):
        device.GetData(n_scans, buffer, buffer_len)
        data = uu.buffer2numpy(buffer, n_scans, n_channels)
        data = uu.reshape_eeg_record(data, n_scans, n_channels)
        print(f'Got data of shape: {data.shape}')

    device.StopAcquisition()


if __name__ == '__main__':

    device = uu.connect_to_unicorn()
    print(f'Version: {device.GetDeviceInformation().DeviceVersion}')
    print(f'{device.GetDeviceInformation().NumberOfEegChannels} channels')

    test_eeg_acquisition(device, 250, 2)
