from dataclasses import dataclass
import numpy as np
import torch
import torchaudio
import pydub
from PIL import Image

from src.torch import torch_utils
from src.audio_data import audio_utils

from typing import Optional


@dataclass(frozen=False)
class SpectrogramParams:
    stereo: bool = False

    # FFT parameters
    sample_rate: int = 44100
    step_size_ms: int = 10
    window_duration_ms: int = 100
    padded_duration_ms: int = 400

    # Mel scale parameters
    num_frequencies: int = 512
    min_frequency: int = 0
    max_frequency: int = 10000
    mel_scale_norm: Optional[str] = None
    mel_scale_type: str = "htk"
    max_mel_iters: int = 200

    # Griffin Lim parameters
    num_griffin_lim_iters: int = 32

    # Image parameterization
    power_for_image: float = 0.25

    @property
    def n_fft(self) -> int:
        return int(self.padded_duration_ms / 1000.0 * self.sample_rate)

    @property
    def win_length(self) -> int:
        return int(self.window_duration_ms / 1000.0 * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.step_size_ms / 1000.0 * self.sample_rate)


class SpectrogramConverter:
    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        self.p = params
        self.device = torch_utils.check_device(device)

        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            win_length=params.win_length,
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
        ).to(self.device)

        self.inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        ).to(self.device)

        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            n_stft=params.n_fft // 2 + 1,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params.n_fft // 2 + 1,
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            max_iter=params.max_mel_iters,
            tolerance_loss=1e-5,
            tolerance_change=1e-8,
            sgdargs=None,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

    def spectrogram_from_audio(self, audio: pydub.AudioSegment) -> np.ndarray:
        assert int(audio.frame_rate) == self.p.sample_rate, "Audio sample rate must match params"
        waveform = np.array([c.get_array_of_samples() for c in audio.split_to_mono()])
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        waveform_tensor = torch.from_numpy(waveform).float().to(self.device)
        amplitudes_mel = self.mel_amplitudes_from_waveform(waveform_tensor)
        return amplitudes_mel.cpu().numpy()

    def audio_from_spectrogram(self, spectrogram: np.ndarray) -> pydub.AudioSegment:
        amplitudes_mel = torch.from_numpy(spectrogram).float().to(self.device)
        waveform = self.waveform_from_mel_amplitudes(amplitudes_mel)
        segment = audio_utils.audio_from_waveform(
            samples=waveform.cpu().numpy(),
            sample_rate=self.p.sample_rate,
            # Normalize the waveform to the range [-1, 1]
            normalize=True,
        )
        return segment

    def mel_amplitudes_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram_complex = self.spectrogram_func(waveform)
        amplitudes = torch.abs(spectrogram_complex)
        return self.mel_scaler(amplitudes)

    def waveform_from_mel_amplitudes(self, amplitudes_mel: torch.Tensor) -> torch.Tensor:
        amplitudes_linear = self.inverse_mel_scaler(amplitudes_mel)
        waveform = self.inverse_spectrogram_func(amplitudes_linear)
        return waveform


class SpectrogramImageConverter:
    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        self.p = params
        self.device = device
        self.converter = SpectrogramConverter(params=params, device=device)

    def spectrogram_image_from_audio(
        self,
        segment: pydub.AudioSegment,
    ) -> Image.Image:
        assert int(segment.frame_rate) == self.p.sample_rate, "Sample rate mismatch"

        if self.p.stereo:
            if segment.channels == 1:
                print("WARNING: Mono audio_data but stereo=True, cloning channel")
                segment = segment.set_channels(2)
            elif segment.channels > 2:
                print("WARNING: Multi channel audio_data, reducing to stereo")
                segment = segment.set_channels(2)
        else:
            if segment.channels > 1:
                print("WARNING: Stereo audio_data but stereo=False, setting to mono")
                segment = segment.set_channels(1)

        spectrogram = self.converter.spectrogram_from_audio(segment)

        image = image_from_spectrogram(
            spectrogram,
            power=self.p.power_for_image,
        )

        return image

    def audio_from_spectrogram_image(self, image: Image.Image, max_value: float = 30e6) -> pydub.AudioSegment:
        spectrogram = spectrogram_from_image(
            image,
            max_value=max_value,
            power=self.p.power_for_image,
            stereo=self.p.stereo,
        )
        segment = self.converter.audio_from_spectrogram(spectrogram)
        return segment


def spectrogram_from_image(image: Image.Image, power: float = 0.25, stereo: bool = False,
                           max_value: float = 30e6) -> np.ndarray:
    # Convert to RGB if single channel
    if image.mode in ("P", "L"):
        image = image.convert("RGB")
    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    # Munge channels into a numpy array of (channels, frequency, time)
    data = np.array(image).transpose(2, 0, 1)
    if stereo:
        # Take the G and B channels as done in image_from_spectrogram
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)
    # Invert
    data = 255 - data
    # Rescale to 0-1
    data = data / 255
    # Reverse the power curve
    data = np.power(data, 1 / power)
    # Rescale to max value
    data = data * max_value

    return data


def image_from_spectrogram(spectrogram: np.ndarray, power: float = 0.25) -> Image.Image:
    # Rescale to 0-1
    max_value = np.max(spectrogram)
    data = spectrogram / max_value
    # Apply the power curve
    data = np.power(data, power)
    # Rescale to 0-255
    data = data * 255
    # Invert
    data = 255 - data
    # Convert to uint8
    data = data.astype(np.uint8)

    # Munge channels into a PIL image
    if data.shape[0] == 1:
        image = Image.fromarray(data[0], mode="L").convert("RGB")
    elif data.shape[0] == 2:
        data = np.array([np.zeros_like(data[0]), data[0], data[1]]).transpose(1, 2, 0)
        image = Image.fromarray(data, mode="RGB")
    else:
        raise NotImplementedError(f"Unsupported number of channels: {data.shape[0]}")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return image