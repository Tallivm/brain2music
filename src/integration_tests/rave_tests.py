import librosa as li

from src.rave.rave_func import load_rave_model, run_rave
from src.audio_data.audio_utils import save_audio_from_wave
from src.constants import SAMPLE_AUDIO_FILEPATH


if __name__ == "__main__":

    for model_name in ['darbouka_onnx', 'vintage', 'nasa']:
        model = load_rave_model(model_name)

        x = li.load(SAMPLE_AUDIO_FILEPATH)[0]
        transformed = run_rave(x, model)

        save_audio_from_wave(transformed, f"../../samples/transformed/sample_music_{model_name}.wav")
