import numpy as np
import torch


def load_rave_model(model_name: str):
    torch.set_grad_enabled(False)
    model = torch.jit.load(f"../../models/{model_name}.ts").eval()
    return model


def run_rave(wave: np.ndarray, rave_model) -> np.ndarray:
    x = torch.from_numpy(wave).reshape(1, 1, -1)
    z = rave_model.encode(x)
    z[:, 0] += torch.linspace(-2, 2, z.shape[-1])  # what happens there?
    y = rave_model.decode(z)
    y = y.numpy().reshape(-1)
    return y
