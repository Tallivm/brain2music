import warnings
import torch


def check_device(device: str, backup: str = "cpu") -> str:
    cuda_not_found = device.lower().startswith("cuda") and not torch.cuda.is_available()
    if cuda_not_found:
        warnings.warn(f"WARNING: {device} is not available, using {backup} instead.", stacklevel=3)
        return backup

    return device
