'use client';

import { motion, useReducedMotion } from 'framer-motion';
import { ArrowRight, Cpu, ListChecks, Sparkles } from 'lucide-react';

const cnnPoints = [
  'Learns intricate acoustic structures directly from spectrograms.',
  'Keeps anomaly scoring stable under noisy, real machine conditions.',
  'Provides the high-sensitivity confidence signal in this demo pipeline.',
];

const lcsPoints = [
  'Converts learned CNN features into readable IF-THEN decision patterns.',
  'Surfaces which evidence trails push a sample toward anomalous behavior.',
  'Adds interpretability that reviewers and engineers can inspect quickly.',
];

export function WhyCnnLcs() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <section id="why-cnn-lcs-section" className="relative overflow-hidden border-t border-green-900/20 px-4 py-14 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.22em] text-green-800">Model Rationale</p>
        <h2 className="font-heading text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
          Why CNN + LCS?
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-700 sm:text-base">
          CNN delivers sharp anomaly sensitivity. LCS delivers human-readable rationale. Side by side, they form a
          high-performance detection system that is also explainable.
        </p>

        <div className="relative mt-8 overflow-hidden rounded-[2rem] border border-green-900/30 bg-[color-mix(in_oklch,var(--background)_72%,white)]/88 p-4 shadow-[0_35px_95px_-65px_rgba(16,101,62,0.72)] sm:p-6 lg:p-8">
          <div className="pointer-events-none absolute inset-0">
            <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(20,95,57,0.11)_1px,transparent_1px),linear-gradient(to_bottom,rgba(20,95,57,0.11)_1px,transparent_1px)] bg-size-[44px_44px]" />
            <motion.div
              className="absolute -left-24 top-2 h-52 w-52 rounded-full bg-green-300/35 blur-3xl"
              animate={prefersReducedMotion ? undefined : { x: ['0%', '220%', '0%'], y: ['0%', '10%', '0%'] }}
              transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
            />
            <motion.div
              className="absolute -right-20 bottom-0 h-52 w-52 rounded-full bg-emerald-300/35 blur-3xl"
              animate={prefersReducedMotion ? undefined : { x: ['0%', '-180%', '0%'], y: ['0%', '-12%', '0%'] }}
              transition={{ duration: 13, repeat: Infinity, ease: 'easeInOut' }}
            />
          </div>

          <div className="relative mb-4 flex items-center justify-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-green-900/25 bg-white/80 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-green-800">
              <Sparkles className="h-3.5 w-3.5" />
              Split-stage architecture: performance + interpretability
            </div>
          </div>

          <div className="relative grid gap-4 lg:grid-cols-[minmax(0,1fr)_76px_minmax(0,1fr)] lg:items-stretch">
            <motion.article
              initial={prefersReducedMotion ? false : { opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.35 }}
              transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
              className="relative overflow-hidden rounded-2xl border border-green-900/25 bg-[linear-gradient(160deg,rgba(255,255,255,0.94)_0%,rgba(223,246,233,0.72)_100%)] p-5 lg:p-6"
            >
              <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_14%_10%,rgba(34,197,94,0.3),transparent_52%)]" />
              <div className="relative">
                <div className="mb-4 flex items-center gap-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-green-800/35 bg-white text-green-800">
                    <Cpu className="h-4 w-4" />
                  </span>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-green-800/80">CNN Side</p>
                    <h3 className="text-lg font-semibold text-slate-900">Performance Engine</h3>
                  </div>
                </div>

                <p className="text-sm leading-relaxed text-slate-700">
                  Deep feature extraction powers anomaly sensitivity across noisy operating conditions.
                </p>

                <ul className="mt-4 space-y-2.5">
                  {cnnPoints.map((point) => (
                    <li key={point} className="flex gap-2.5 text-sm leading-relaxed text-slate-700">
                      <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-green-700" />
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.article>

            <div className="relative hidden items-center justify-center lg:flex">
              <div className="absolute inset-y-5 left-1/2 w-px -translate-x-1/2 bg-green-800/35" />
              <motion.div
                className="relative inline-flex h-10 w-10 items-center justify-center rounded-full border border-green-800/45 bg-white text-green-800"
                animate={prefersReducedMotion ? undefined : { y: [0, -8, 0], scale: [1, 1.05, 1] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
              >
                <ArrowRight className="h-4 w-4" />
              </motion.div>
            </div>

            <motion.article
              initial={prefersReducedMotion ? false : { opacity: 0, x: 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.35 }}
              transition={{ duration: 0.55, delay: prefersReducedMotion ? 0 : 0.08, ease: [0.22, 1, 0.36, 1] }}
              className="relative overflow-hidden rounded-2xl border border-green-900/25 bg-[linear-gradient(200deg,rgba(255,255,255,0.94)_0%,rgba(206,249,233,0.74)_100%)] p-5 lg:p-6"
            >
              <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_88%_12%,rgba(16,185,129,0.34),transparent_52%)]" />
              <div className="relative">
                <div className="mb-4 flex items-center gap-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-green-800/35 bg-white text-green-800">
                    <ListChecks className="h-4 w-4" />
                  </span>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-green-800/80">LCS Side</p>
                    <h3 className="text-lg font-semibold text-slate-900">Interpretability Engine</h3>
                  </div>
                </div>

                <p className="text-sm leading-relaxed text-slate-700">
                  Rule extraction translates learned behavior into inspectable decision rationale.
                </p>

                <ul className="mt-4 space-y-2.5">
                  {lcsPoints.map((point) => (
                    <li key={point} className="flex gap-2.5 text-sm leading-relaxed text-slate-700">
                      <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-700" />
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.article>
          </div>

          <motion.div
            initial={prefersReducedMotion ? false : { opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.4 }}
            transition={{ duration: 0.45, delay: prefersReducedMotion ? 0 : 0.18, ease: [0.22, 1, 0.36, 1] }}
            className="relative mt-4 rounded-xl border border-green-900/25 bg-white/85 px-4 py-3"
          >
            <p className="text-sm font-medium text-slate-700">
              <span className="font-semibold text-slate-900">Combined effect:</span> CNN provides high-quality anomaly scoring,
              while LCS exposes rule-level evidence so decisions remain auditable.
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
