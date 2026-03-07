import { useTranslations } from "next-intl";
import { setRequestLocale } from "next-intl/server";
import CryAnalyzer from "@/components/CryAnalyzer";
import LanguageSwitcher from "@/components/LanguageSwitcher";

type Props = {
  params: Promise<{ locale: string }>;
};

export default async function Home({ params }: Props) {
  const { locale } = await params;
  setRequestLocale(locale);

  return <HomeContent />;
}

function HomeContent() {
  const t = useTranslations();

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-foreground/5 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="text-lg font-bold text-foreground">
              {t("nav.title")}
            </h1>
            <p className="text-xs text-muted">{t("nav.subtitle")}</p>
          </div>
          <LanguageSwitcher />
        </div>
      </header>

      {/* Main */}
      <main className="flex flex-1 flex-col">
        {/* Hero section */}
        <section className="px-4 pt-12 pb-8 text-center sm:pt-20 sm:pb-12">
          <div className="mx-auto max-w-2xl">
            <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-5xl">
              {t("hero.heading")}
            </h2>
            <p className="mb-3 text-base text-muted sm:text-lg">
              {t("hero.description")}
            </p>
            <p className="text-xs font-medium text-primary">
              {t("hero.powered")}
            </p>
          </div>
        </section>

        {/* Feature badges */}
        <section className="px-4 pb-8">
          <div className="mx-auto grid max-w-2xl grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl bg-surface p-3 text-center shadow-sm">
              <div className="mb-1 text-2xl">
                <svg className="mx-auto h-6 w-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
              </div>
              <p className="text-xs font-semibold text-foreground">{t("hero.features.edgeAi")}</p>
              <p className="mt-0.5 text-[10px] leading-tight text-muted">{t("hero.features.edgeAiDesc")}</p>
            </div>
            <div className="rounded-xl bg-surface p-3 text-center shadow-sm">
              <div className="mb-1 text-2xl">
                <svg className="mx-auto h-6 w-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              </div>
              <p className="text-xs font-semibold text-foreground">{t("hero.features.zeroLatency")}</p>
              <p className="mt-0.5 text-[10px] leading-tight text-muted">{t("hero.features.zeroLatencyDesc")}</p>
            </div>
            <div className="rounded-xl bg-surface p-3 text-center shadow-sm">
              <div className="mb-1 text-2xl">
                <svg className="mx-auto h-6 w-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M18.364 5.636a9 9 0 11-12.728 0M12 9v4" /><circle cx="12" cy="17" r="1" fill="currentColor" /></svg>
              </div>
              <p className="text-xs font-semibold text-foreground">{t("hero.features.offline")}</p>
              <p className="mt-0.5 text-[10px] leading-tight text-muted">{t("hero.features.offlineDesc")}</p>
            </div>
            <div className="rounded-xl bg-surface p-3 text-center shadow-sm">
              <div className="mb-1 text-2xl">
                <svg className="mx-auto h-6 w-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
              </div>
              <p className="text-xs font-semibold text-foreground">{t("hero.features.privacy")}</p>
              <p className="mt-0.5 text-[10px] leading-tight text-muted">{t("hero.features.privacyDesc")}</p>
            </div>
          </div>
        </section>

        {/* Analyzer section */}
        <section className="flex flex-1 items-start justify-center px-4 py-8 sm:py-12">
          <CryAnalyzer />
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-foreground/5 py-6 text-center text-xs text-muted">
        <p>{t("footer.disclaimer")}</p>
      </footer>
    </div>
  );
}
