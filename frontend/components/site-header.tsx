'use client';

import { ExternalLink } from 'lucide-react';

const navLinks = [
  { href: '#upload-section', label: 'Upload' },
  { href: '#how-it-works-section', label: 'Flow' },
  { href: '#why-cnn-lcs-section', label: 'CNN + LCS' },
  { href: '#limitations-section', label: 'Limitations' },
  { href: '#tech-stack-section', label: 'Stack' },
  { href: '#background-research-section', label: 'Background' },
];

type SiteHeaderProps = {
  githubUrl: string;
};

export function SiteHeader({ githubUrl }: SiteHeaderProps) {
  return (
    <header className="absolute inset-x-0 top-0 z-40 bg-transparent py-3 backdrop-blur-[2px]">
      <div className="mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 lg:grid lg:grid-cols-3 lg:items-center lg:gap-4">
          <div className="min-w-0">
            <p className="font-heading text-xl font-semibold tracking-wide text-slate-900 transition-all duration-300">
              CNN-LCS
            </p>
            <p className="text-sm text-slate-600 transition-all duration-300">
              Engine Sound Anomaly Detection
            </p>
          </div>

          <nav className="flex flex-wrap items-center justify-center gap-2">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="inline-flex shrink-0 items-center rounded-full border border-green-800/35 bg-white/80 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.08em] text-slate-800 transition hover:border-green-900 hover:bg-green-100"
              >
                {link.label}
              </a>
            ))}
          </nav>

          <a
            href={githubUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex w-fit items-center gap-2 rounded-full border border-green-800/45 bg-white/85 px-4 py-2.5 text-sm font-semibold text-slate-900 transition hover:border-green-900 hover:bg-green-100 lg:ml-auto"
          >
            <ExternalLink className="h-4 w-4" />
            GitHub
          </a>
        </div>
      </div>
    </header>
  );
}
