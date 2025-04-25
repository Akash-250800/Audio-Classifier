
import torch
import librosa
import numpy as np
from cnn_model import SpectrogramCNN


def audio_to_mel(file, sr=22050, n_mels=128, max_len=660):
    y, _ = librosa.load(file, sr=sr, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize and resize
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = mel_db[np.newaxis, np.newaxis, :, :]  # Shape: [1, 1, H, W]
    return torch.tensor(mel_db, dtype=torch.float32)

def load_model(path, num_classes):
    from cnn_model import SpectrogramCNN  # or DeeperCNN if using that
    model = SpectrogramCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
