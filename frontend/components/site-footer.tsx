type SiteFooterProps = {
  githubUrl: string;
};

export function SiteFooter({ githubUrl }: SiteFooterProps) {
  return (
    <footer className="border-t border-green-900/20 bg-white/85 px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-7xl flex-col items-center gap-3 text-sm text-slate-600 sm:grid sm:grid-cols-3">
        <p>
          CNN-LCS © 2026 — Engine Sound Anomaly Detection
        </p>
        <p className="text-center">
          © Vagdevi Ponugupati, Donna Mathew, Juan Rea
        </p>
        <a
          href={githubUrl}
          target="_blank"
          rel="noreferrer"
          className="font-semibold text-green-800 transition hover:text-green-700 sm:ml-auto"
        >
          View source on GitHub
        </a>
      </div>
    </footer>
  );
}
