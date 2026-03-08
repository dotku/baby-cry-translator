"""
Train a CNN model on DonateACry + Mendeley corpora using MFCC features (PyTorch).
Uses GroupKFold cross-validation to prevent data leakage between babies.
Exports to ONNX for browser inference via onnxruntime-web.

Optimized: MFCC extracted once per original file, augmentation at MFCC level.

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
from sklearn.model_selection import GroupKFold
from collections import Counter
import json
import time

# === Config ===
DONATEACRY_DIR = "/tmp/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
MENDELEY_DIR = "/tmp/mendeley-cry"
MODEL_OUTPUT_DIR = "/Users/wlin/dev/baby-cry-translator/public/model"
CATEGORIES = ["hungry", "tired", "discomfort", "belly_pain", "burping"]
CATEGORY_LABELS = {cat: i for i, cat in enumerate(CATEGORIES)}
SR = 22050
DURATION = 4
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Pre-compute mel filterbank once
_FB_CACHE = None


def mel_filterbank(sr, n_fft, n_mels):
    global _FB_CACHE
    if _FB_CACHE is not None:
        return _FB_CACHE
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
    _FB_CACHE = filterbank
    return filterbank


def extract_mfcc(audio, sr):
    _, _, Zxx = stft(audio, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    power_spectrum = np.abs(Zxx) ** 2
    fb = mel_filterbank(sr, N_FFT, N_MELS)
    mel_spec = np.dot(fb, power_spectrum)
    mel_spec = np.log(mel_spec + 1e-9)
    mfcc = dct(mel_spec, type=2, axis=0, norm="ortho")[:N_MFCC]
    return mfcc.astype(np.float32)


def augment_mfcc(mfcc, n_augments=5):
    """Augment at MFCC level — much faster than audio-level augmentation."""
    augmented = [mfcc.copy()]
    n_mfcc, n_frames = mfcc.shape

    for _ in range(n_augments - 1):
        aug = mfcc.copy()

        # Random gain scaling
        if np.random.random() > 0.3:
            aug = aug * np.random.uniform(0.8, 1.2)

        # Add noise to MFCC coefficients
        if np.random.random() > 0.3:
            aug = aug + np.random.randn(*aug.shape) * np.random.uniform(0.05, 0.2)

        # Frequency masking (SpecAugment)
        if np.random.random() > 0.3:
            f = np.random.randint(1, min(8, n_mfcc))
            f0 = np.random.randint(0, n_mfcc - f)
            aug[f0:f0 + f, :] = 0

        # Time masking (SpecAugment)
        if np.random.random() > 0.3:
            t = np.random.randint(1, min(30, n_frames))
            t0 = np.random.randint(0, n_frames - t)
            aug[:, t0:t0 + t] = 0

        # Time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-20, 20)
            aug = np.roll(aug, shift, axis=1)

        augmented.append(aug)
    return augmented


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


def extract_baby_id(filename, source):
    """Extract baby/speaker ID for group-based splitting."""
    if source == "donateacry":
        parts = filename.split("-")
        if len(parts) >= 5:
            return f"dac_{parts[0][:8]}"
        return f"dac_{filename[:8]}"
    elif source == "mendeley":
        parts = filename.split("_")
        if len(parts) >= 2:
            return f"men_{parts[1]}"
        return f"men_{filename[:8]}"
    return filename[:8]


def load_dataset():
    """Load dataset from multiple sources. Returns raw MFCCs + labels + groups."""
    category_mfccs = {cat: [] for cat in CATEGORIES}  # list of (mfcc, baby_id)

    # 1. Load DonateACry corpus
    print("   Loading DonateACry corpus...")
    t0 = time.time()
    for category in CATEGORIES:
        dir_path = os.path.join(DONATEACRY_DIR, category)
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
        for fname in files:
            try:
                audio = load_audio(os.path.join(dir_path, fname))
                mfcc = extract_mfcc(audio, SR)
                baby_id = extract_baby_id(fname, "donateacry")
                category_mfccs[category].append((mfcc, baby_id))
            except Exception as e:
                print(f"     Error: {fname}: {e}")
    print(f"   DonateACry loaded in {time.time()-t0:.1f}s")

    # 2. Load Mendeley corpus
    print("   Loading Mendeley corpus...")
    t0 = time.time()
    mendeley_map = {"hungry": "hungry", "discomfort": "discomfort"}
    for mend_cat, our_cat in mendeley_map.items():
        dir_path = os.path.join(MENDELEY_DIR, mend_cat)
        if not os.path.isdir(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
        for fname in files:
            try:
                audio = load_audio(os.path.join(dir_path, fname))
                mfcc = extract_mfcc(audio, SR)
                baby_id = extract_baby_id(fname, "mendeley")
                category_mfccs[our_cat].append((mfcc, baby_id))
            except Exception as e:
                print(f"     Error: {fname}: {e}")
    print(f"   Mendeley loaded in {time.time()-t0:.1f}s")

    # Report raw counts
    raw_counts = {cat: len(samples) for cat, samples in category_mfccs.items()}
    print(f"   Raw dataset: {raw_counts} (total: {sum(raw_counts.values())})")
    for cat in CATEGORIES:
        baby_ids = set(bid for _, bid in category_mfccs[cat])
        print(f"   {cat}: {len(category_mfccs[cat])} samples from {len(baby_ids)} babies")

    # Balance classes with MFCC-level augmentation
    max_count = max(raw_counts.values())
    X, y, groups = [], [], []

    print("   Augmenting at MFCC level...")
    t0 = time.time()
    for category in CATEGORIES:
        samples = category_mfccs[category]
        label = CATEGORY_LABELS[category]
        n_raw = len(samples)
        if n_raw == 0:
            continue
        augs_per_sample = max(3, int(np.ceil(max_count / n_raw)))

        for mfcc, baby_id in samples:
            for aug_mfcc in augment_mfcc(mfcc, n_augments=augs_per_sample):
                X.append(aug_mfcc)
                y.append(label)
                groups.append(baby_id)
    print(f"   Augmentation done in {time.time()-t0:.1f}s")

    final_counts = Counter(y)
    print(f"   After augmentation: { {CATEGORIES[k]: v for k, v in sorted(final_counts.items())} }")
    print(f"   Total samples: {len(X)}, Unique babies: {len(set(groups))}")

    return np.array(X), np.array(y), np.array(groups)


class CryClassifierCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_fold(X_train, y_train, X_test, y_test, input_shape, fold_num=0):
    """Train one fold and return test accuracy and model state."""
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = CryClassifierCNN(input_shape, len(CATEGORIES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_acc = 0
    best_state = None
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

        if (epoch + 1) % 10 == 0:
            print(f"     Fold {fold_num} Epoch {epoch+1:3d}: loss={avg_loss:.4f} acc={acc:.1%}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"     Fold {fold_num} early stop at epoch {epoch+1}, best acc={best_acc:.1%}")
                break

    return best_acc, best_state


def train():
    print("=" * 60)
    print("BabyTalk - Training MFCC + CNN v2 (PyTorch)")
    print("Multi-source data + GroupKFold cross-validation")
    print("=" * 60)
    total_start = time.time()

    # Load data
    print("\n1. Loading datasets...")
    X, y, groups = load_dataset()

    # Add channel dim: (N, 1, n_mfcc, time_frames)
    X = X[:, np.newaxis, :, :]
    input_shape = (X.shape[2], X.shape[3])

    # GroupKFold cross-validation
    print("\n2. Cross-validation (GroupKFold, 5 folds)...")
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    fold_accs = []
    best_overall_acc = 0
    best_overall_state = None

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_babies = set(groups[train_idx])
        test_babies = set(groups[test_idx])
        print(f"\n   Fold {fold+1}: train={len(X_train)} ({len(train_babies)} babies), "
              f"test={len(X_test)} ({len(test_babies)} babies)")

        acc, state = train_one_fold(X_train, y_train, X_test, y_test, input_shape, fold + 1)
        fold_accs.append(acc)

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_state = state

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n3. Cross-validation results:")
    for i, acc in enumerate(fold_accs):
        print(f"   Fold {i+1}: {acc:.1%}")
    print(f"   Mean: {mean_acc:.1%} ± {std_acc:.1%}")

    # Final model: train on all data
    print("\n4. Training final model on all data...")
    all_ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    all_loader = DataLoader(all_ds, batch_size=32, shuffle=True)

    model = CryClassifierCNN(input_shape, len(CATEGORIES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    for epoch in range(80):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for xb, yb in all_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: loss={train_loss/len(all_loader):.4f} train_acc={correct/total:.1%}")

    # Export to ONNX
    print("\n5. Exporting to ONNX...")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.eval()
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
        "categoryMap": {"hungry": "hungry", "tired": "tired", "discomfort": "discomfort",
                        "belly_pain": "belly_pain", "burping": "burp"},
        "sampleRate": SR,
        "duration": DURATION,
        "nMfcc": N_MFCC,
        "nFft": N_FFT,
        "hopLength": HOP_LENGTH,
        "nMels": N_MELS,
        "inputShape": list(input_shape),
        "accuracy": float(mean_acc),
        "accuracy_std": float(std_acc),
        "cv_folds": n_splits,
        "data_sources": ["donateacry", "mendeley"],
    }
    config_path = os.path.join(MODEL_OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Config saved: {config_path}")

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed/60:.1f} min! CV accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    train()
