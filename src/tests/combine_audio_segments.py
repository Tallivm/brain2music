from src.data.utils import combine_pydub_audio_from_folder


if __name__ == "__main__":
    combined = combine_pydub_audio_from_folder('../../samples/outputs/uncombined/')
    combined.export("../../samples/outputs/combined.mp3", format="mp3")
