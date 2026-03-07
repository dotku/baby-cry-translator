"use client";

import { useState, useRef, useCallback } from "react";
import { useTranslations } from "next-intl";
import RecordButton from "./RecordButton";
import WaveformVisualizer from "./WaveformVisualizer";
import ResultCard from "./ResultCard";
import { AudioRecorderUtil } from "@/lib/audio-utils";
import { classifyCry, type ClassificationResult } from "@/lib/cry-classifier";

type Status = "idle" | "recording" | "analyzing" | "done" | "error";

export default function CryAnalyzer() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef(new AudioRecorderUtil());
  const t = useTranslations("recorder");

  const handleToggleRecording = useCallback(async () => {
    if (status === "recording") {
      // Stop recording
      setStatus("analyzing");
      try {
        const { audioData, sampleRate } = await recorderRef.current.stopRecording();
        setAnalyserNode(null);
        const classification = await classifyCry(audioData, sampleRate);
        setResult(classification);
        setStatus("done");
      } catch {
        setError(t("error"));
        setStatus("error");
        recorderRef.current.cleanup();
      }
    } else {
      // Start recording
      setResult(null);
      setError(null);
      try {
        const analyser = await recorderRef.current.startRecording();
        setAnalyserNode(analyser);
        setStatus("recording");
      } catch {
        setError(t("permission"));
        setStatus("error");
      }
    }
  }, [status, t]);

  const handleRecordAgain = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
    recorderRef.current.cleanup();
    recorderRef.current = new AudioRecorderUtil();
  }, []);

  return (
    <div className="flex flex-col items-center gap-8">
      {status !== "done" && (
        <>
          <RecordButton
            status={status}
            onClick={handleToggleRecording}
          />
          <WaveformVisualizer
            analyserNode={analyserNode}
            isRecording={status === "recording"}
          />
        </>
      )}

      {error && (
        <div className="rounded-xl bg-red-50 p-4 text-center text-red-600 dark:bg-red-900/20 dark:text-red-400">
          <p>{error}</p>
          <button
            onClick={handleRecordAgain}
            className="mt-2 text-sm font-medium underline"
          >
            {t("tryAgain")}
          </button>
        </div>
      )}

      {status === "done" && result && (
        <ResultCard result={result} onRecordAgain={handleRecordAgain} />
      )}
    </div>
  );
}
