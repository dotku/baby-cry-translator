"""
Train a CNN model on DonateACry corpus using MFCC features (PyTorch).
Exports to ONNX for browser inference via onnxruntime-web.

Usage: python3 scripts/train_model.py
"""

import os
import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.fftpack import dct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
import json

# === Config ===
CORPUS_DIR = "/tmp/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
MODEL_OUTPUT_DIR = "/Users/wlin/dev/baby-cry-translator/public/model"
CATEGORIES = ["hungry", "tired", "discomfort", "belly_pain", "burping"]
CATEGORY_LABELS = {cat: i for i, cat in enumerate(CATEGORIES)}
SR = 22050
DURATION = 4
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


def mel_filterbank(sr, n_fft, n_mels):
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            if j < n_freqs:
                filterbank[i, j] = (j - bin_points[i]) / max(bin_points[i + 1] - bin_points[i], 1)
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            if j < n_freqs:
                filterbank[i, j] = (bin_points[i + 2] - j) / max(bin_points[i + 2] - bin_points[i + 1], 1)
    return filterbank


def extract_mfcc(audio, sr):
    _, _, Zxx = stft(audio, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    power_spectrum = np.abs(Zxx) ** 2
    fb = mel_filterbank(sr, N_FFT, N_MELS)
    mel_spec = np.dot(fb, power_spectrum)
    mel_spec = np.log(mel_spec + 1e-9)
    mfcc = dct(mel_spec, type=2, axis=0, norm="ortho")[:N_MFCC]
    return mfcc.astype(np.float32)


def load_audio(filepath):
    audio, file_sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        ratio = SR / file_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    target_length = SR * DURATION
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def augment_audio(audio, n_augments=5):
    """Generate augmented versions of audio."""
    augmented = [audio.copy()]
    for _ in range(n_augments - 1):
        aug = audio.copy()
        # Random combination of augmentations
        if np.random.random() > 0.5:
            aug = aug + np.random.randn(len(aug)) * np.random.uniform(0.002, 0.01)
        if np.random.random() > 0.5:
            shift = int(SR * np.random.uniform(-0.3, 0.3))
            aug = np.roll(aug, shift)
        if np.random.random() > 0.5:
            aug = aug * np.random.uniform(0.7, 1.3)
        if np.random.random() > 0.5:
            # Pitch-like: resample slightly
            factor = np.random.uniform(0.9, 1.1)
            indices = np.linspace(0, len(aug) - 1, int(len(aug) * factor))
            indices = np.clip(indices, 0, len(aug) - 1)
            aug_resampled = np.interp(np.arange(len(aug)),
                                       np.linspace(0, len(aug) - 1, len(indices)),
                                       np.interp(indices, np.arange(len(aug)), aug))
            aug = aug_resampled[:len(aug)]
        augmented.append(aug)
    return augmented


def load_dataset():
    """Load dataset with oversampling to balance classes."""
    X, y = [], []
    raw_counts = {}
    category_audios = {}

    # First pass: load all audio
    for category in CATEGORIES:
        dir_path = os.path.join(CORPUS_DIR, category)
        files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
        raw_counts[category] = len(files)
        category_audios[category] = []
        for fname in files:
            filepath = os.path.join(dir_path, fname)
            try:
                audio = load_audio(filepath)
                category_audios[category].append(audio)
            except Exception as e:
                print(f"  Error: {fname}: {e}")

    # Target: balance all classes to ~same total samples
    max_count = max(raw_counts.values())  # 382 (hungry)
    target_per_class = max_count  # aim for equal representation

    for category in CATEGORIES:
        audios = category_audios[category]
        label = CATEGORY_LABELS[category]
        n_raw = len(audios)
        # How many augmented samples per original to reach target
        augs_per_sample = max(5, int(np.ceil(target_per_class / n_raw)))

        for audio in audios:
            for aug in augment_audio(audio, n_augments=augs_per_sample):
                mfcc = extract_mfcc(aug, SR)
                X.append(mfcc)
                y.append(label)

    # Report
    final_counts = Counter(y)
    print(f"Raw dataset: {raw_counts}")
    print(f"After oversampling: { {CATEGORIES[k]: v for k, v in sorted(final_counts.items())} }")
    print(f"Total samples: {len(X)}")

    print(f"Dataset: {raw_counts}")
    print(f"Total samples (augmented): {len(X)}")
    return np.array(X), np.array(y)


class CryClassifierCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train():
    print("=" * 60)
    print("BabyTalk - Training MFCC + CNN (PyTorch)")
    print("=" * 60)

    # Load data
    print("\n1. Loading dataset...")
    X, y = load_dataset()

    # Add channel dim: (N, 1, n_mfcc, time_frames)
    X = X[:, np.newaxis, :, :]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Model
    input_shape = (X_train.shape[2], X_train.shape[3])
    model = CryClassifierCNN(input_shape, len(CATEGORIES))
    print(f"\n2. Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    # Train
    print("\n3. Training...")
    best_acc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate
        model.eval()
        correct, total_samples = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total_samples += len(yb)

        acc = correct / total_samples
        avg_loss = train_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0 or acc > best_acc:
            print(f"   Epoch {epoch+1:3d}: loss={avg_loss:.4f} acc={acc:.1%}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    # Final evaluation
    print(f"\n4. Best test accuracy: {best_acc:.1%}")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            all_preds.extend(out.argmax(dim=1).numpy())
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n   Per-class results:")
    for i, cat in enumerate(CATEGORIES):
        mask = all_labels == i
        if mask.sum() > 0:
            cat_acc = (all_preds[mask] == i).mean()
            print(f"   {cat:15s}: {cat_acc:.1%} ({mask.sum()} samples)")

    print("\n   Confusion Matrix:")
    for i, cat in enumerate(CATEGORIES):
        row = [str(((all_labels == i) & (all_preds == j)).sum()).rjust(4) for j in range(len(CATEGORIES))]
        print(f"   {cat:15s}: {''.join(row)}")
    print(f"   {'':15s}  {''.join(c[:4].rjust(4) for c in CATEGORIES)}")

    # Export to ONNX
    print("\n5. Exporting to ONNX...")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    dummy_input = torch.randn(1, 1, input_shape[0], input_shape[1])
    onnx_path = os.path.join(MODEL_OUTPUT_DIR, "cry_model.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["audio_mfcc"],
        output_names=["probabilities"],
        dynamic_axes={"audio_mfcc": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=13,
    )
    print(f"   ONNX model saved: {onnx_path}")
    print(f"   Model size: {os.path.getsize(onnx_path) / 1024:.0f} KB")

    # Save config
    config = {
        "categories": CATEGORIES,
        "categoryMap": {"hungry": "hungry", "tired": "tired", "discomfort": "discomfort", "belly_pain": "belly_pain", "burping": "burp"},
        "sampleRate": SR,
        "duration": DURATION,
        "nMfcc": N_MFCC,
        "nFft": N_FFT,
        "hopLength": HOP_LENGTH,
        "nMels": N_MELS,
        "inputShape": list(input_shape),
        "accuracy": float(best_acc),
    }
    config_path = os.path.join(MODEL_OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Config saved: {config_path}")

    print("\n" + "=" * 60)
    print(f"Done! Best accuracy: {best_acc:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    train()
