/**
 * Benchmark: test cry-classifier against labeled samples from donateacry corpus
 *
 * Usage: npx tsx scripts/benchmark.mts
 */

import * as fs from "fs";
import * as path from "path";
import wavDecoder from "wav-decoder";

// --- Inline classifier (same logic as src/lib/cry-classifier.ts) ---

type CryCategory = "hungry" | "tired" | "discomfort" | "belly_pain" | "burp";

interface ClassificationResult {
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

const CATEGORY_PROFILES = {
  hungry: {
    lowFreqWeight: 0.7,
    midFreqWeight: 0.5,
    highFreqWeight: 0.2,
    rhythmWeight: 0.8,
  },
  tired: {
    lowFreqWeight: 0.5,
    midFreqWeight: 0.6,
    highFreqWeight: 0.3,
    rhythmWeight: 0.4,
  },
  discomfort: {
    lowFreqWeight: 0.3,
    midFreqWeight: 0.7,
    highFreqWeight: 0.5,
    rhythmWeight: 0.5,
  },
  belly_pain: {
    lowFreqWeight: 0.2,
    midFreqWeight: 0.4,
    highFreqWeight: 0.8,
    rhythmWeight: 0.3,
  },
  burp: {
    lowFreqWeight: 0.6,
    midFreqWeight: 0.3,
    highFreqWeight: 0.4,
    rhythmWeight: 0.6,
  },
};

function extractFeatures(audioData: Float32Array, sampleRate: number) {
  const frameSize = 1024;
  const numFrames = Math.floor(audioData.length / frameSize);

  let totalEnergy = 0;
  let lowFreqEnergy = 0;
  let midFreqEnergy = 0;
  let highFreqEnergy = 0;
  let zeroCrossings = 0;

  for (let i = 1; i < audioData.length; i++) {
    if (
      (audioData[i] >= 0 && audioData[i - 1] < 0) ||
      (audioData[i] < 0 && audioData[i - 1] >= 0)
    ) {
      zeroCrossings++;
    }
  }
  const zcr = zeroCrossings / audioData.length;

  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * frameSize;
    let frameEnergy = 0;

    for (let i = 0; i < frameSize; i++) {
      const sample = audioData[start + i] || 0;
      frameEnergy += sample * sample;
    }
    totalEnergy += frameEnergy;

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

  const totalBandEnergy = lowFreqEnergy + midFreqEnergy + highFreqEnergy || 1;

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
  const rhythmScore =
    1 / (1 + energyVariance / (meanEnergy * meanEnergy + 1e-6));

  return {
    lowFreqRatio: lowFreqEnergy / totalBandEnergy,
    midFreqRatio: midFreqEnergy / totalBandEnergy,
    highFreqRatio: highFreqEnergy / totalBandEnergy,
    zcr,
    rhythmScore,
    rmsEnergy: Math.sqrt(totalEnergy / (audioData.length || 1)),
  };
}

function classifyCry(
  audioData: Float32Array,
  sampleRate: number
): ClassificationResult {
  const features = extractFeatures(audioData, sampleRate);

  const scores: Record<CryCategory, number> = {} as Record<
    CryCategory,
    number
  >;

  for (const category of CATEGORIES) {
    const profile = CATEGORY_PROFILES[category];
    const score =
      (1 - Math.abs(features.lowFreqRatio - profile.lowFreqWeight * 0.5)) *
        0.25 +
      (1 - Math.abs(features.midFreqRatio - profile.midFreqWeight * 0.5)) *
        0.25 +
      (1 - Math.abs(features.highFreqRatio - profile.highFreqWeight * 0.5)) *
        0.25 +
      (1 - Math.abs(features.rhythmScore - profile.rhythmWeight)) * 0.25;

    const pitchFactor =
      category === "belly_pain"
        ? features.zcr * 2
        : category === "hungry"
          ? (1 - features.zcr) * 1.5
          : 1;

    scores[category] = Math.max(0, Math.min(1, score * pitchFactor));
  }

  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
  for (const cat of CATEGORIES) {
    scores[cat] = scores[cat] / totalScore;
  }

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

// --- Benchmark logic ---

const CORPUS_DIR =
  "/tmp/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data";

const LABEL_MAP: Record<string, CryCategory> = {
  hungry: "hungry",
  tired: "tired",
  discomfort: "discomfort",
  belly_pain: "belly_pain",
  burping: "burp",
};

const SAMPLES_PER_CATEGORY = 10;

interface TestResult {
  file: string;
  expected: CryCategory;
  predicted: CryCategory;
  confidence: number;
  correct: boolean;
  allScores: Record<CryCategory, number>;
}

async function loadWav(
  filePath: string
): Promise<{ audioData: Float32Array; sampleRate: number }> {
  const buffer = fs.readFileSync(filePath);
  const decoded = await wavDecoder.decode(buffer);
  return {
    audioData: decoded.channelData[0],
    sampleRate: decoded.sampleRate,
  };
}

async function runBenchmark() {
  console.log("=".repeat(70));
  console.log("BabyTalk Classifier Benchmark");
  console.log(
    `Testing ${SAMPLES_PER_CATEGORY} samples per category (${Object.keys(LABEL_MAP).length * SAMPLES_PER_CATEGORY} total)`
  );
  console.log("=".repeat(70));
  console.log();

  const results: TestResult[] = [];

  for (const [dirName, expectedCategory] of Object.entries(LABEL_MAP)) {
    const dirPath = path.join(CORPUS_DIR, dirName);
    const files = fs
      .readdirSync(dirPath)
      .filter((f) => f.endsWith(".wav"))
      .slice(0, SAMPLES_PER_CATEGORY);

    for (const file of files) {
      const filePath = path.join(dirPath, file);
      try {
        const { audioData, sampleRate } = await loadWav(filePath);
        const result = classifyCry(audioData, sampleRate);

        const testResult: TestResult = {
          file: `${dirName}/${file.slice(0, 20)}...`,
          expected: expectedCategory,
          predicted: result.category,
          confidence: result.confidence,
          correct: result.category === expectedCategory,
          allScores: result.allScores,
        };

        results.push(testResult);

        const icon = testResult.correct ? "✓" : "✗";
        console.log(
          `${icon} [${expectedCategory.padEnd(11)}] → predicted: ${result.category.padEnd(11)} (${result.confidence}%)  ${file.slice(0, 40)}`
        );
      } catch (err) {
        console.log(`  SKIP ${file} - ${(err as Error).message}`);
      }
    }
  }

  // Summary
  console.log();
  console.log("=".repeat(70));
  console.log("SUMMARY");
  console.log("=".repeat(70));

  const total = results.length;
  const correct = results.filter((r) => r.correct).length;
  const accuracy = ((correct / total) * 100).toFixed(1);

  console.log(`Total samples:  ${total}`);
  console.log(`Correct:        ${correct}`);
  console.log(`Accuracy:       ${accuracy}%`);
  console.log();

  // Per-category breakdown
  console.log("Per-category accuracy:");
  for (const [dirName, expectedCategory] of Object.entries(LABEL_MAP)) {
    const catResults = results.filter((r) => r.expected === expectedCategory);
    const catCorrect = catResults.filter((r) => r.correct).length;
    const catAcc =
      catResults.length > 0
        ? ((catCorrect / catResults.length) * 100).toFixed(0)
        : "N/A";
    console.log(
      `  ${expectedCategory.padEnd(12)}: ${catCorrect}/${catResults.length} (${catAcc}%)`
    );
  }

  // Confusion matrix
  console.log();
  console.log("Confusion Matrix (rows=expected, cols=predicted):");
  const header = ["", ...CATEGORIES.map((c) => c.slice(0, 7).padEnd(7))].join(
    " | "
  );
  console.log(header);
  console.log("-".repeat(header.length));

  for (const expected of CATEGORIES) {
    const row = CATEGORIES.map((predicted) => {
      const count = results.filter(
        (r) => r.expected === expected && r.predicted === predicted
      ).length;
      return String(count).padEnd(7);
    });
    console.log([expected.slice(0, 7).padEnd(7), ...row].join(" | "));
  }

  console.log();
  console.log("=".repeat(70));
  console.log(
    `Overall Accuracy: ${accuracy}% (${correct}/${total}) — heuristic-based classifier`
  );
  console.log(
    "Note: This is a demo classifier using spectral heuristics, not a trained model."
  );
  console.log("=".repeat(70));
}

runBenchmark().catch(console.error);
