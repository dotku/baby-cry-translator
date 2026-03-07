"use client";

import { useTranslations } from "next-intl";

interface Props {
  status: "idle" | "recording" | "analyzing" | "done" | "error";
  onClick: () => void;
}

export default function RecordButton({ status, onClick }: Props) {
  const t = useTranslations("recorder");

  const isRecording = status === "recording";
  const isAnalyzing = status === "analyzing";

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative">
        {isRecording && (
          <div className="animate-pulse-ring absolute inset-0 rounded-full bg-accent/30" />
        )}
        <button
          onClick={onClick}
          disabled={isAnalyzing}
          className={`relative z-10 flex h-28 w-28 items-center justify-center rounded-full shadow-lg transition-all sm:h-32 sm:w-32 ${
            isRecording
              ? "bg-accent scale-110 shadow-accent/30"
              : isAnalyzing
                ? "bg-muted cursor-not-allowed"
                : "bg-primary hover:bg-primary-light hover:scale-105 active:scale-95"
          }`}
        >
          {isRecording ? (
            <svg
              className="h-10 w-10 text-white"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          ) : isAnalyzing ? (
            <svg
              className="h-10 w-10 animate-spin text-white"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
          ) : (
            <svg
              className="h-10 w-10 text-white"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
          )}
        </button>
      </div>

      <p className="text-sm font-medium text-muted">
        {isRecording
          ? t("recording")
          : isAnalyzing
            ? t("analyzing")
            : t("idle")}
      </p>
    </div>
  );
}
