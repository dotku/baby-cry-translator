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
