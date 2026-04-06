'use client';

import { useEffect, useState } from 'react';
import { ArrowDownCircle, ExternalLink } from 'lucide-react';

type SiteHeaderProps = {
  githubUrl: string;
};

export function SiteHeader({ githubUrl }: SiteHeaderProps) {
  const [compact, setCompact] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setCompact(window.scrollY > 36);
    };

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className={`fixed left-0 right-0 top-0 z-40 bg-transparent transition-all duration-300 ${
        compact ? 'py-1.5' : 'py-3'
      }`}
    >
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div>
          <p className={`font-heading font-semibold tracking-wide text-slate-900 transition-all duration-300 ${compact ? 'text-base' : 'text-xl'}`}>
            CNN-LCS
          </p>
          <p className={`text-slate-600 transition-all duration-300 ${compact ? 'text-xs' : 'text-sm'}`}>
            Engine Sound Anomaly Detection
          </p>
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          <a
            href={githubUrl}
            target="_blank"
            rel="noreferrer"
            className={`inline-flex items-center gap-2 rounded-full border border-green-800/45 bg-white text-slate-900 transition hover:border-green-900 hover:bg-green-100 ${
              compact ? 'px-3 py-2 text-xs font-semibold' : 'px-4 py-2.5 text-sm font-semibold'
            }`}
          >
            <ExternalLink className={`${compact ? 'h-3.5 w-3.5' : 'h-4 w-4'}`} />
            GitHub
          </a>
          <a
            href="#upload-section"
            className={`inline-flex items-center gap-2 rounded-full bg-green-700 text-white transition hover:bg-green-600 ${
              compact ? 'px-3 py-2 text-xs font-semibold' : 'px-4 py-2.5 text-sm font-semibold'
            }`}
          >
            <ArrowDownCircle className={`${compact ? 'h-3.5 w-3.5' : 'h-4 w-4'}`} />
            Go to Upload
          </a>
        </div>
      </div>
    </header>
  );
}
