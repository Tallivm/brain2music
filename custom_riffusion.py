from dataclasses import dataclass
import numpy as np
import torch
import torchaudio
import pydub

import torch_utils
import audio_utils

from typing import Optional


@dataclass(frozen=True)
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
