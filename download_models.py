# download_models.py

from faster_whisper import WhisperModel
import torch

def download_models():
    print("Downloading Whisper transcription model ('base')...")
    WhisperModel(model_size_or_path="base", device="cpu")

    print("Downloading Whisper realtime transcription model ('tiny')...")
    WhisperModel(model_size_or_path="tiny", device="cpu")

    print("Downloading Silero VAD model ('snakers4/silero-vad')...")
    torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
