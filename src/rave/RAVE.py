import torch
import librosa as li
import soundfile as sf

# https://github.com/acids-ircam/RAVE/discussions/208
# pre-rained model available, to be placed in folder models : https://acids-ircam.github.io/rave_models_download

model_name = "darbouka_onnx"  # "darbouka_onnx" #"nasa"
input_wav_name = "sample_music" #"jazzfunk_sample"  # "some_audio"

torch.set_grad_enabled(False)

print("Loading model...")
model = torch.jit.load(f"models/{model_name}.ts").eval()
# other models : nasa.ts, vintage.ts

print("Loading sample...")
x = li.load(f"data/{input_wav_name}.wav")[0]

x = torch.from_numpy(x).reshape(1,1,-1)

print("Encoding...")
z = model.encode(x)

z[:, 0] += torch.linspace(-2,2,z.shape[-1])

print("Decoding...")
y = model.decode(z).numpy().reshape(-1)

print("Saving output...")
sf.write(f"output/{input_wav_name}_{model_name}_out.wav", y, 44100)

print("Done")