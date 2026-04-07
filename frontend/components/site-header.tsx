'use client';

import { ExternalLink, Menu, X } from 'lucide-react';
import { useState } from 'react';

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
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <header className="absolute inset-x-0 top-0 z-40 bg-transparent py-3 backdrop-blur-[2px]">
      <div className="mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 lg:grid lg:grid-cols-3 lg:items-center lg:gap-4">
          <div className="flex min-w-0 items-start justify-between gap-3 lg:block">
            <div>
              <p className="font-heading text-xl font-semibold tracking-wide text-slate-900 transition-all duration-300">
                CNN-LCS
              </p>
              <p className="text-sm text-slate-600 transition-all duration-300">
                Engine Sound Anomaly Detection
              </p>
            </div>
            <button
              type="button"
              onClick={() => setIsMobileMenuOpen((open) => !open)}
              aria-expanded={isMobileMenuOpen}
              aria-controls="mobile-site-nav"
              aria-label={isMobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-green-800/45 bg-white/85 text-slate-900 transition hover:border-green-900 hover:bg-green-100 lg:hidden"
            >
              {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>

          <nav className="hidden flex-wrap items-center justify-center gap-2 lg:flex">
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

          <nav
            id="mobile-site-nav"
            className={`${
              isMobileMenuOpen ? 'flex' : 'hidden'
            } flex-col gap-2 lg:hidden`}
          >
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className="inline-flex w-fit items-center rounded-full border border-green-800/35 bg-white/80 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.08em] text-slate-800 transition hover:border-green-900 hover:bg-green-100"
              >
                {link.label}
              </a>
            ))}
            <a
              href={githubUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex w-fit items-center gap-2 rounded-full border border-green-800/45 bg-white/85 px-4 py-2.5 text-sm font-semibold text-slate-900 transition hover:border-green-900 hover:bg-green-100"
            >
              <ExternalLink className="h-4 w-4" />
              GitHub
            </a>
          </nav>

          <a
            href={githubUrl}
            target="_blank"
            rel="noreferrer"
            className="hidden w-fit items-center gap-2 rounded-full border border-green-800/45 bg-white/85 px-4 py-2.5 text-sm font-semibold text-slate-900 transition hover:border-green-900 hover:bg-green-100 lg:ml-auto lg:inline-flex"
          >
            <ExternalLink className="h-4 w-4" />
            GitHub
          </a>
        </div>
      </div>
    </header>
  );
}
