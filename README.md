# brain2music
A Python and SD pipeline to convert EEG data to music in real time

## Pipeline Overview

```mermaid
graph TB
    A[EEG] --> C[EEG Features]
    C -->D[Raw Spectrogram]
    D -->|Riffusion| E[Transformed Spectrogram]
    E -->|Torch| F[Wave]
    F -->H[Audio]
    H -->I[Play]
    
    subgraph Transformations 
        D
        E
        F
    end
```

## Setup

### Install packages

```commandline
pip install -r requirements.txt
```

### Install PyTorch

1. Go to PyTorch website https://pytorch.org/.
1. Scroll down to installation instructions.
1. Select Stable build, your OS, package manages (ex.: Pip), language=Python, CUDA or CPU. 
1. Copy the installation command and run it in a shell/cmd in root project folder

You can check if you have CUDA by running in cmd and looking in the top right corner for "CUDA Version":

```commandline
nvidia-smi
```

### Setup Rave (optional)

Download models from: https://acids-ircam.github.io/rave_models_download.

Put them into `/models`.