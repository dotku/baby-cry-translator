import * as ort from "onnxruntime-web";

export type CryCategory =
  | "hungry"
  | "uncomfortable"
  | "fussy";

export interface ClassificationResult {
  category: CryCategory;
  confidence: number;
  allScores: Record<CryCategory, number>;
}

// Must match training parameters in train_model.py
const SR = 22050;
const DURATION = 4;
const N_MFCC = 40;
const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_MELS = 128;
const INPUT_FRAMES = 174;

const INDEX_TO_CATEGORY: CryCategory[] = [
  "hungry",
  "uncomfortable",
  "fussy",
];

let session: ort.InferenceSession | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!session) {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("/model/cry_model.onnx", {
      executionProviders: ["wasm"],
    });
  }
  return session;
}

function melFilterbank(): Float32Array[] {
  const fMax = SR / 2;
  const melMin = 2595.0 * Math.log10(1.0);
  const melMax = 2595.0 * Math.log10(1.0 + fMax / 700.0);

  const melPoints: number[] = [];
  for (let i = 0; i <= N_MELS + 1; i++) {
    melPoints.push(melMin + (i * (melMax - melMin)) / (N_MELS + 1));
  }

  const hzPoints = melPoints.map((m) => 700.0 * (Math.pow(10, m / 2595.0) - 1.0));
  const binPoints = hzPoints.map((hz) => Math.floor(((N_FFT + 1) * hz) / SR));
  const nFreqs = Math.floor(N_FFT / 2) + 1;
  const fb: Float32Array[] = [];

  for (let i = 0; i < N_MELS; i++) {
    const filter = new Float32Array(nFreqs);
    for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
      if (j < nFreqs)
        filter[j] = (j - binPoints[i]) / Math.max(binPoints[i + 1] - binPoints[i], 1);
    }
    for (let j = binPoints[i + 1]; j < binPoints[i + 2]; j++) {
      if (j < nFreqs)
        filter[j] = (binPoints[i + 2] - j) / Math.max(binPoints[i + 2] - binPoints[i + 1], 1);
    }
    fb.push(filter);
  }
  return fb;
}

function hannWindow(): Float32Array {
  const w = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N_FFT - 1)));
  }
  return w;
}

function fft(real: Float32Array, imag: Float32Array): { real: Float32Array; imag: Float32Array } {
  const n = real.length;
  const outR = new Float32Array(n);
  const outI = new Float32Array(n);
  const bits = Math.log2(n);

  for (let i = 0; i < n; i++) {
    let rev = 0;
    for (let j = 0; j < bits; j++) rev = (rev << 1) | ((i >> j) & 1);
    outR[rev] = real[i];
    outI[rev] = imag[i];
  }

  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const angle = (-2 * Math.PI) / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < half; j++) {
        const cos = Math.cos(angle * j);
        const sin = Math.sin(angle * j);
        const tR = outR[i + j + half] * cos - outI[i + j + half] * sin;
        const tI = outR[i + j + half] * sin + outI[i + j + half] * cos;
        outR[i + j + half] = outR[i + j] - tR;
        outI[i + j + half] = outI[i + j] - tI;
        outR[i + j] += tR;
        outI[i + j] += tI;
      }
    }
  }
  return { real: outR, imag: outI };
}

function extractMfcc(audioData: Float32Array, sampleRate: number): Float32Array {
  // Resample to target SR
  let audio = audioData;
  if (sampleRate !== SR) {
    const ratio = SR / sampleRate;
    const newLen = Math.floor(audio.length * ratio);
    const resampled = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const src = i / ratio;
      const idx = Math.floor(src);
      const frac = src - idx;
      resampled[i] = (audio[idx] || 0) * (1 - frac) + (audio[idx + 1] || 0) * frac;
    }
    audio = resampled;
  }

  // Pad or truncate
  const targetLen = SR * DURATION;
  if (audio.length < targetLen) {
    const padded = new Float32Array(targetLen);
    padded.set(audio);
    audio = padded;
  } else {
    audio = audio.slice(0, targetLen);
  }

  // Normalize
  let maxVal = 0;
  for (let i = 0; i < audio.length; i++) maxVal = Math.max(maxVal, Math.abs(audio[i]));
  if (maxVal > 0) for (let i = 0; i < audio.length; i++) audio[i] /= maxVal;

  // STFT
  const window = hannWindow();
  const nFreqs = Math.floor(N_FFT / 2) + 1;
  const numFrames = Math.floor((audio.length - N_FFT) / HOP_LENGTH) + 1;
  const fftSize = Math.pow(2, Math.ceil(Math.log2(N_FFT)));

  const powerSpec: Float32Array[] = [];
  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * HOP_LENGTH;
    const real = new Float32Array(fftSize);
    const imag = new Float32Array(fftSize);
    for (let i = 0; i < N_FFT; i++) real[i] = (audio[start + i] || 0) * window[i];

    const result = fft(real, imag);
    const power = new Float32Array(nFreqs);
    for (let i = 0; i < nFreqs; i++)
      power[i] = result.real[i] * result.real[i] + result.imag[i] * result.imag[i];
    powerSpec.push(power);
  }

  // Mel filterbank
  const fb = melFilterbank();
  const melSpec: Float32Array[] = [];
  for (let frame = 0; frame < numFrames; frame++) {
    const mel = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      for (let f = 0; f < nFreqs; f++) sum += fb[m][f] * powerSpec[frame][f];
      mel[m] = Math.log(sum + 1e-9);
    }
    melSpec.push(mel);
  }

  // DCT-II → MFCC
  const mfcc = new Float32Array(N_MFCC * INPUT_FRAMES);
  for (let frame = 0; frame < INPUT_FRAMES; frame++) {
    const src = frame < numFrames ? frame : numFrames - 1;
    for (let k = 0; k < N_MFCC; k++) {
      let sum = 0;
      for (let n = 0; n < N_MELS; n++)
        sum += melSpec[src][n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N_MELS));
      const norm = k === 0 ? Math.sqrt(1 / N_MELS) : Math.sqrt(2 / N_MELS);
      mfcc[k * INPUT_FRAMES + frame] = sum * norm;
    }
  }

  return mfcc;
}

function softmax(arr: Float32Array): Float32Array {
  const max = Math.max(...arr);
  const exp = new Float32Array(arr.length);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    exp[i] = Math.exp(arr[i] - max);
    sum += exp[i];
  }
  for (let i = 0; i < arr.length; i++) exp[i] /= sum;
  return exp;
}

export async function classifyCry(
  audioData: Float32Array,
  sampleRate: number
): Promise<ClassificationResult> {
  const sess = await getSession();
  const mfcc = extractMfcc(audioData, sampleRate);

  const tensor = new ort.Tensor("float32", mfcc, [1, 1, N_MFCC, INPUT_FRAMES]);
  const results = await sess.run({ audio_mfcc: tensor });
  const logits = results.probabilities.data as Float32Array;
  const probs = softmax(logits);

  const allScores: Record<CryCategory, number> = {} as Record<CryCategory, number>;
  let bestIdx = 0;
  let bestScore = 0;

  for (let i = 0; i < INDEX_TO_CATEGORY.length; i++) {
    allScores[INDEX_TO_CATEGORY[i]] = probs[i];
    if (probs[i] > bestScore) {
      bestScore = probs[i];
      bestIdx = i;
    }
  }

  return {
    category: INDEX_TO_CATEGORY[bestIdx],
    confidence: Math.round(bestScore * 100),
    allScores,
  };
}
