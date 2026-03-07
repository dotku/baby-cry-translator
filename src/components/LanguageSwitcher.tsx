"use client";

import { useLocale } from "next-intl";
import { useRouter, usePathname } from "@/i18n/routing";

export default function LanguageSwitcher() {
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  const toggleLocale = () => {
    const next = locale === "en" ? "zh" : "en";
    router.replace(pathname, { locale: next });
  };

  return (
    <button
      onClick={toggleLocale}
      className="rounded-full border border-foreground/20 px-3 py-1.5 text-sm font-medium transition-colors hover:bg-foreground/5"
    >
      {locale === "en" ? "中文" : "EN"}
    </button>
  );
}
