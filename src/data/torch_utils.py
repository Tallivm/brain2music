from dataclasses import dataclass
import numpy as np
import torch
import torchaudio
import pydub

from src.constants import AUDIO_SAMPLE_RATE, MIN_AUDIO_FREQUENCY, MAX_AUDIO_FREQUENCY, SPECTROGRAM_HEIGHT
from src.data.utils import produce_audio_from_wave, postprocess_wave

from typing import Optional


def check_device(device: str, backup: str = "cpu") -> str:
    cuda_not_found = device.lower().startswith("cuda") and not torch.cuda.is_available()
    if cuda_not_found:
        print(f"WARNING: {device} is not available, using {backup} instead.")
        return backup
    if device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        return backup
    return device


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """
    Helper function to spherically interpolate two arrays v1 v2.
    """
    inputs_are_torch = False
    input_device = None
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > dot_threshold:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


@dataclass(frozen=False)
class SpectrogramParams:

    # FFT parameters
    sample_rate: int = AUDIO_SAMPLE_RATE
    step_size_ms: int = 10
    window_duration_ms: int = 100
    padded_duration_ms: int = 400

    # Mel scale parameters
    num_frequencies: int = SPECTROGRAM_HEIGHT
    min_frequency: int = MIN_AUDIO_FREQUENCY
    max_frequency: int = MAX_AUDIO_FREQUENCY
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
    def __init__(self, params: Optional[SpectrogramParams] = None, device: str = "cuda"):
        if params is None:
            params = SpectrogramParams()
        self.p = params
        self.device = check_device(device)

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

    def spectrogram_from_audio(self, audio: pydub.AudioSegment, use_mel: bool = True) -> np.ndarray:
        assert int(audio.frame_rate) == self.p.sample_rate, "Audio sample rate must match params"
        waveform = np.array([c.get_array_of_samples() for c in audio.split_to_mono()])
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        waveform_tensor = torch.from_numpy(waveform).float().to(self.device)
        amplitudes = self.amplitudes_from_waveform(waveform_tensor, use_mel)
        return amplitudes.cpu().numpy()

    def wave_from_spectrogram(self, spectrogram: np.ndarray, use_mel: bool = True) -> np.ndarray:
        amplitudes = torch.from_numpy(spectrogram).float().to(self.device)
        waveform = self.waveform_from_amplitudes(amplitudes, use_mel)
        return waveform.cpu().numpy().squeeze()

    def audio_from_spectrogram(self, spectrogram: np.ndarray, use_mel: bool = True) -> pydub.AudioSegment:
        wave = self.wave_from_spectrogram(spectrogram, use_mel)
        wave = postprocess_wave(wave)
        return produce_audio_from_wave(wave)

    def amplitudes_from_waveform(self, waveform: torch.Tensor, use_mel: bool = True) -> torch.Tensor:
        spectrogram_complex = self.spectrogram_func(waveform)
        amplitudes = torch.abs(spectrogram_complex)
        if use_mel:
            return self.mel_scaler(amplitudes)
        else:
            return amplitudes

    def waveform_from_amplitudes(self, amplitudes: torch.Tensor, use_mel: bool = True) -> torch.Tensor:
        if use_mel:
            with torch.enable_grad():
                amplitudes = self.inverse_mel_scaler(amplitudes)
        return self.inverse_spectrogram_func(amplitudes)
