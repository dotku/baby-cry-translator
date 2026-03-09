"use client";

import { useTranslations } from "next-intl";
import type { ClassificationResult, CryCategory } from "@/lib/cry-classifier";

interface Props {
  result: ClassificationResult;
  onRecordAgain: () => void;
}

const CATEGORY_ICONS: Record<CryCategory, string> = {
  hungry: "🍼",
  uncomfortable: "😣",
  fussy: "😫",
};

const CATEGORY_COLORS: Record<CryCategory, string> = {
  hungry: "from-orange-400 to-amber-500",
  uncomfortable: "from-rose-400 to-pink-500",
  fussy: "from-indigo-400 to-blue-500",
};

export default function ResultCard({ result, onRecordAgain }: Props) {
  const t = useTranslations("results");
  const tc = useTranslations("categories");
  const tr = useTranslations("recorder");

  const sortedCategories = Object.entries(result.allScores)
    .sort(([, a], [, b]) => b - a)
    .map(([cat, score]) => ({ category: cat as CryCategory, score }));

  return (
    <div className="mx-auto w-full max-w-md space-y-6">
      {/* Main result */}
      <div
        className={`rounded-2xl bg-gradient-to-br ${CATEGORY_COLORS[result.category]} p-6 text-white shadow-xl`}
      >
        <div className="mb-2 text-center text-4xl">
          {CATEGORY_ICONS[result.category]}
        </div>
        <h3 className="text-center text-2xl font-bold">
          {tc(`${result.category}.label`)}
        </h3>
        <p className="mt-1 text-center text-lg font-semibold opacity-90">
          {t("confidence")}: {result.confidence}%
        </p>
      </div>

      {/* Tip */}
      <div className="rounded-xl bg-surface p-4 shadow-md">
        <h4 className="mb-1 text-sm font-semibold uppercase tracking-wide text-muted">
          {t("tip")}
        </h4>
        <p className="text-foreground">
          {tc(`${result.category}.tip`)}
        </p>
      </div>

      {/* All scores */}
      <div className="rounded-xl bg-surface p-4 shadow-md">
        <h4 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
          {t("title")}
        </h4>
        <div className="space-y-2">
          {sortedCategories.map(({ category, score }) => (
            <div key={category} className="flex items-center gap-3">
              <span className="w-6 text-center text-lg">
                {CATEGORY_ICONS[category]}
              </span>
              <span className="w-20 text-sm font-medium">
                {tc(`${category}.label`)}
              </span>
              <div className="flex-1">
                <div className="h-2.5 overflow-hidden rounded-full bg-foreground/10">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${CATEGORY_COLORS[category]} transition-all duration-700`}
                    style={{ width: `${Math.round(score * 100)}%` }}
                  />
                </div>
              </div>
              <span className="w-10 text-right text-sm text-muted">
                {Math.round(score * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <p className="text-center text-xs text-muted">{t("disclaimer")}</p>

      {/* Record again */}
      <div className="text-center">
        <button
          onClick={onRecordAgain}
          className="rounded-full bg-primary px-6 py-2.5 font-medium text-white transition-colors hover:bg-primary-light"
        >
          {tr("recordAgain")}
        </button>
      </div>
    </div>
  );
}
