"""
Benchmark the trained ONNX model against the DonateACry corpus.
Usage: python3 scripts/benchmark_onnx.py
"""

import os
import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.fftpack import dct
import onnxruntime as ort
from collections import Counter

CORPUS_DIR = "/tmp/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
MODEL_PATH = "/Users/wlin/dev/baby-cry-translator/public/model/cry_model.onnx"
CATEGORIES = ["hungry", "tired", "discomfort", "belly_pain", "burping"]
CATEGORY_DISPLAY = {"hungry": "hungry", "tired": "tired", "discomfort": "discomfort", "belly_pain": "belly_pain", "burping": "burp"}
SR = 22050
DURATION = 4
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SAMPLES_PER_CATEGORY = 10


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
    power = np.abs(Zxx) ** 2
    fb = mel_filterbank(sr, N_FFT, N_MELS)
    mel_spec = np.dot(fb, power)
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


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    print("=" * 70)
    print("BabyTalk ONNX Model Benchmark")
    print("=" * 70)

    sess = ort.InferenceSession(MODEL_PATH)
    input_name = sess.get_inputs()[0].name

    results = []

    for category in CATEGORIES:
        dir_path = os.path.join(CORPUS_DIR, category)
        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".wav")])

        # Use ALL files for a complete benchmark
        for fname in files[:SAMPLES_PER_CATEGORY]:
            filepath = os.path.join(dir_path, fname)
            try:
                audio = load_audio(filepath)
                mfcc = extract_mfcc(audio, SR)
                input_data = mfcc[np.newaxis, np.newaxis, :, :]  # [1, 1, 40, frames]

                # Pad/truncate frames to 174
                target_frames = 174
                if input_data.shape[3] < target_frames:
                    pad_width = ((0, 0), (0, 0), (0, 0), (0, target_frames - input_data.shape[3]))
                    input_data = np.pad(input_data, pad_width)
                else:
                    input_data = input_data[:, :, :, :target_frames]

                logits = sess.run(None, {input_name: input_data})[0][0]
                probs = softmax(logits)
                pred_idx = np.argmax(probs)
                pred_cat = CATEGORIES[pred_idx]
                confidence = probs[pred_idx] * 100

                correct = pred_cat == category
                icon = "OK" if correct else "XX"
                results.append((category, pred_cat, correct, confidence))

                print(f"  {icon} [{category:12s}] -> {pred_cat:12s} ({confidence:5.1f}%)  {fname[:45]}")

            except Exception as e:
                print(f"  SKIP {fname}: {e}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results)
    correct = sum(1 for _, _, c, _ in results if c)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"Total samples:  {total}")
    print(f"Correct:        {correct}")
    print(f"Accuracy:       {accuracy:.1f}%")
    print()

    print("Per-category accuracy:")
    for cat in CATEGORIES:
        cat_results = [(p, c) for e, p, c, _ in results if e == cat]
        cat_correct = sum(1 for _, c in cat_results if c)
        cat_total = len(cat_results)
        cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
        print(f"  {cat:15s}: {cat_correct}/{cat_total} ({cat_acc:.0f}%)")

    # Confusion matrix
    print()
    print("Confusion Matrix:")
    header = "               " + "".join(c[:7].rjust(8) for c in CATEGORIES)
    print(header)
    for i, cat in enumerate(CATEGORIES):
        row = []
        for j, pred_cat in enumerate(CATEGORIES):
            count = sum(1 for e, p, _, _ in results if e == cat and p == pred_cat)
            row.append(str(count).rjust(8))
        print(f"  {cat:13s}{''.join(row)}")

    print()
    print(f"Overall: {accuracy:.1f}% ({correct}/{total})")
    print("=" * 70)


if __name__ == "__main__":
    main()
