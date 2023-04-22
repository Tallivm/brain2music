import numpy as np

# EEG constants ---------------
MIN_EEG_FREQUENCY = 0.5
MAX_EEG_FREQUENCY = 40
N_EEG_FREQUENCIES = 512
EEG_FREQUENCIES = np.linspace(MIN_EEG_FREQUENCY, MAX_EEG_FREQUENCY, N_EEG_FREQUENCIES)
SAMPLE_RATE = 250
SEGMENT_LEN_S = 5
CHANNEL_IDS = (0, 1, 2, 3)

# Spectrogram constants -------
SPECTROGRAM_WIDTH = 512
SPECTROGRAM_HEIGHT = 512
SPECTROGRAM_MAX_VALUE = 30e6

# Model constants -------------
RIFFUSION_CHECKPOINT = "riffusion/riffusion-model-v1"
SCHEDULER_OPTIONS = [
    "DPMSolverMultistepScheduler",
    "PNDMScheduler",
    "DDIMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
]
TEXT_PROMPT = 'dreamy tune'
TEXT_NEGATIVE_PROMPT = None
DENOISING_STRENGTH = 0.65
GUIDANCE_SCALE = 7.0
INFERENCE_STEPS = 25

# Audio constants -------------
AUDIO_SAMPLE_RATE = 44100
MIN_AUDIO_FREQUENCY = 0
MAX_AUDIO_FREQUENCY = 5000
PLAYBACK_SLEEP_TIME_S = 4.98
