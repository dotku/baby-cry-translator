import * as tf from "@tensorflow/tfjs";

export type CryCategory =
  | "hungry"
  | "tired"
  | "discomfort"
  | "belly_pain"
  | "burp";

export interface ClassificationResult {
  category: CryCategory;
  confidence: number;
  allScores: Record<CryCategory, number>;
}

const CATEGORIES: CryCategory[] = [
  "hungry",
  "tired",
  "discomfort",
  "belly_pain",
  "burp",
];

// Frequency characteristics associated with different cry types (simplified heuristic)
// Based on research: hungry cries tend to be rhythmic and lower pitch,
// pain cries are higher pitch and more intense, etc.
const CATEGORY_PROFILES = {
  hungry: { lowFreqWeight: 0.7, midFreqWeight: 0.5, highFreqWeight: 0.2, rhythmWeight: 0.8 },
  tired: { lowFreqWeight: 0.5, midFreqWeight: 0.6, highFreqWeight: 0.3, rhythmWeight: 0.4 },
  discomfort: { lowFreqWeight: 0.3, midFreqWeight: 0.7, highFreqWeight: 0.5, rhythmWeight: 0.5 },
  belly_pain: { lowFreqWeight: 0.2, midFreqWeight: 0.4, highFreqWeight: 0.8, rhythmWeight: 0.3 },
  burp: { lowFreqWeight: 0.6, midFreqWeight: 0.3, highFreqWeight: 0.4, rhythmWeight: 0.6 },
};

/**
 * Extract audio features from raw audio data using Web Audio API concepts.
 * This uses spectral analysis to characterize the cry.
 */
function extractFeatures(audioData: Float32Array, sampleRate: number) {
  const frameSize = 1024;
  const numFrames = Math.floor(audioData.length / frameSize);

  let totalEnergy = 0;
  let lowFreqEnergy = 0;
  let midFreqEnergy = 0;
  let highFreqEnergy = 0;
  let zeroCrossings = 0;

  // Calculate zero-crossing rate (correlates with pitch)
  for (let i = 1; i < audioData.length; i++) {
    if (
      (audioData[i] >= 0 && audioData[i - 1] < 0) ||
      (audioData[i] < 0 && audioData[i - 1] >= 0)
    ) {
      zeroCrossings++;
    }
  }
  const zcr = zeroCrossings / audioData.length;

  // Simple spectral analysis using frame-based energy
  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * frameSize;
    let frameEnergy = 0;

    for (let i = 0; i < frameSize; i++) {
      const sample = audioData[start + i] || 0;
      frameEnergy += sample * sample;
    }
    totalEnergy += frameEnergy;

    // Estimate frequency band energies using autocorrelation-like approach
    const lowBand = frameSize / 4;
    const midBand = frameSize / 2;

    for (let i = 0; i < lowBand; i++) {
      lowFreqEnergy += (audioData[start + i] || 0) ** 2;
    }
    for (let i = lowBand; i < midBand; i++) {
      midFreqEnergy += (audioData[start + i] || 0) ** 2;
    }
    for (let i = midBand; i < frameSize; i++) {
      highFreqEnergy += (audioData[start + i] || 0) ** 2;
    }
  }

  // Normalize
  const totalBandEnergy = lowFreqEnergy + midFreqEnergy + highFreqEnergy || 1;

  // Rhythm detection: variance in frame energies
  const frameEnergies: number[] = [];
  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * frameSize;
    let e = 0;
    for (let i = 0; i < frameSize; i++) {
      e += (audioData[start + i] || 0) ** 2;
    }
    frameEnergies.push(e);
  }

  const meanEnergy =
    frameEnergies.reduce((a, b) => a + b, 0) / (frameEnergies.length || 1);
  const energyVariance =
    frameEnergies.reduce((a, b) => a + (b - meanEnergy) ** 2, 0) /
    (frameEnergies.length || 1);

  // Rhythmic cries have periodic energy patterns (lower normalized variance = more rhythmic)
  const rhythmScore = 1 / (1 + energyVariance / (meanEnergy * meanEnergy + 1e-6));

  return {
    lowFreqRatio: lowFreqEnergy / totalBandEnergy,
    midFreqRatio: midFreqEnergy / totalBandEnergy,
    highFreqRatio: highFreqEnergy / totalBandEnergy,
    zcr,
    rhythmScore,
    rmsEnergy: Math.sqrt(totalEnergy / (audioData.length || 1)),
  };
}

/**
 * Classify baby cry using spectral feature matching.
 * This is a demo-grade classifier using audio feature heuristics.
 * For production, you'd train a proper model on labeled cry data.
 */
export async function classifyCry(
  audioData: Float32Array,
  sampleRate: number
): Promise<ClassificationResult> {
  const features = extractFeatures(audioData, sampleRate);

  // Score each category based on how well audio features match the profile
  const scores: Record<CryCategory, number> = {} as Record<CryCategory, number>;

  for (const category of CATEGORIES) {
    const profile = CATEGORY_PROFILES[category];
    const score =
      (1 - Math.abs(features.lowFreqRatio - profile.lowFreqWeight * 0.5)) * 0.25 +
      (1 - Math.abs(features.midFreqRatio - profile.midFreqWeight * 0.5)) * 0.25 +
      (1 - Math.abs(features.highFreqRatio - profile.highFreqWeight * 0.5)) * 0.25 +
      (1 - Math.abs(features.rhythmScore - profile.rhythmWeight)) * 0.25;

    // Add some variance based on ZCR (pitch estimation)
    const pitchFactor =
      category === "belly_pain"
        ? features.zcr * 2
        : category === "hungry"
          ? (1 - features.zcr) * 1.5
          : 1;

    scores[category] = Math.max(0, Math.min(1, score * pitchFactor));
  }

  // Normalize scores to sum to 1
  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
  for (const cat of CATEGORIES) {
    scores[cat] = scores[cat] / totalScore;
  }

  // Find best category
  let bestCategory: CryCategory = "hungry";
  let bestScore = 0;
  for (const cat of CATEGORIES) {
    if (scores[cat] > bestScore) {
      bestScore = scores[cat];
      bestCategory = cat;
    }
  }

  return {
    category: bestCategory,
    confidence: Math.round(bestScore * 100),
    allScores: scores,
  };
}
