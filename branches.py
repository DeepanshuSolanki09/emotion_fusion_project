import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import DropBlock2d

N_CLASSES = 3


class EEGFNN(nn.Module):
    def __init__(self, in_features=750, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())


class FaceCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1, 64), conv_bn_relu(64, 64), nn.MaxPool2d(2),
            conv_bn_relu(64, 128), conv_bn_relu(128, 128), nn.MaxPool2d(2),
            conv_bn_relu(128, 256), conv_bn_relu(256, 256),
            DropBlock2d(p=0.1, block_size=3), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(9216, 256), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class SpeechCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1, 32), nn.MaxPool2d(2),
            conv_bn_relu(32, 64), nn.MaxPool2d(2),
            conv_bn_relu(64, 128), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def predict_proba(model, x):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).cpu().numpy()


class VideoBranch:
    def __init__(self, device="cpu"):
        import torchvision
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.cnn = torchvision.models.resnet18(weights=w)
        self.cnn.fc = nn.Identity()
        self.cnn.eval().to(device)
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.device = device
        self.clf = make_pipeline(StandardScaler(),
                                 SVC(kernel="rbf", C=10, probability=True, random_state=42))

    @torch.no_grad()
    def visual_features(self, frames):
        f = self.cnn(frames.to(self.device)).cpu().numpy()
        return np.concatenate([f.mean(0), f.std(0)])

    @staticmethod
    def audio_features(wav_path, sr=22050):
        import librosa
        y, _ = librosa.load(wav_path, sr=sr)
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.concatenate([m.mean(1), m.std(1)])

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def speech_to_mfcc_map(wav_path, sr=22050, dur=3.0, n_frames=128):
    import librosa
    y, _ = librosa.load(wav_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y / (np.abs(y).max() + 1e-9)
    n = int(sr * dur)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512,
                             n_mels=64, window="hamming")
    m = m[:, :n_frames]
    return ((m - m.mean()) / (m.std() + 1e-9)).astype(np.float32)[None]


def eeg_fft_features(window, fs, n_features=750):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [1, 45], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, window, axis=-1) * np.hanning(window.shape[-1])
    f = np.abs(np.fft.rfft(x, axis=-1)).ravel()
    return np.pad(f, (0, max(0, n_features - len(f))))[:n_features].astype(np.float32)
