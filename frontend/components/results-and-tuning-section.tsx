'use client';

import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';

/* ================================================================
   DATA — every metric from the original section is preserved here
   ================================================================ */

type ModelConfig = {
  name: string;
  short: string;
  color: string;
  metrics: Record<string, number>;
};

/* Attempt 1 — three model configurations charted on F1/Precision/Recall/AUC */
const attempt1Models: ModelConfig[] = [
  {
    name: 'CNN @ threshold 0.50',
    short: 'CNN t = 0.50',
    color: '#34d399',
    metrics: { F1: 0.4036, Precision: 0.4125, Recall: 0.395, AUC: 0.7963 },
  },
  {
    name: 'CNN @ threshold 0.67 (tuned)',
    short: 'CNN t = 0.67',
    color: '#86efac',
    metrics: { F1: 0.408, Precision: 0.4751, Recall: 0.3575, AUC: 0.7963 },
  },
  {
    name: 'LCS on CNN features (k = 50 / 1280)',
    short: 'LCS k = 50',
    color: '#5eead4',
    metrics: { F1: 0.3805, Precision: 0.464, Recall: 0.3225, AUC: 0.7347 },
  },
];

const chartMetrics = ['F1', 'Precision', 'Recall', 'AUC'];

/* Attempt 2 — F1 side-by-side */
const attempt2Items = [
  { label: 'CNN reference', value: 0.408, color: '#34d399' },
  { label: 'Best LCS sweep', value: 0.4046, color: '#fbbf24' },
];

/* Prioritised next steps */
const levers = [
  {
    n: 1,
    title: 'Better CNN training (highest impact)',
    text: 'Train to ~30 epochs and add SpecAugment so LCS gets stronger features.',
  },
  {
    n: 2,
    title: 'Threshold search as a first-class step',
    text: 'Keep validating threshold policy; anomaly-heavy recall settings can still improve F1 trade-offs.',
  },
  {
    n: 3,
    title: 'Longer LCS search budget',
    text: 'Increase ExSTraCS iterations to 100,000 with N = 3000 for rare anomaly rules.',
  },
  {
    n: 4,
    title: 'LCS representation capacity',
    text: 'Raise selected features from k = 50 toward k = 100–150 and retune mutation/hyperparameters.',
  },
];

/* ================================================================
   GROUPED BAR CHART — pure SVG, framer-motion animated
   ================================================================ */

const CW = 520;
const CH = 290;
const PAD = { top: 30, right: 16, bottom: 54, left: 46 };
const IW = CW - PAD.left - PAD.right;
const IH = CH - PAD.top - PAD.bottom;
const Y_MAX = 1.0;
const Y_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

function GroupedBarChart({
  models,
  metrics,
}: {
  models: ModelConfig[];
  metrics: string[];
}) {
  const groupCount = metrics.length;
  const barCount = models.length;
  const groupWidth = IW / groupCount;
  const barWidth = 30;
  const barGap = 4;
  const totalBarsW = barCount * barWidth + (barCount - 1) * barGap;

  return (
    <div className="mt-4">
      <svg
        viewBox={`0 0 ${CW} ${CH}`}
        className="w-full h-auto"
        role="img"
        aria-label="Grouped bar chart comparing model configurations across F1, Precision, Recall, and AUC"
      >
        {/* Y-axis grid */}
        {Y_TICKS.map((tick) => {
          const y = PAD.top + IH - (tick / Y_MAX) * IH;
          return (
            <g key={tick}>
              <line
                x1={PAD.left}
                y1={y}
                x2={CW - PAD.right}
                y2={y}
                stroke="rgba(16,185,129,0.10)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 8}
                y={y + 3.5}
                textAnchor="end"
                fill="#64748b"
                fontSize={10}
              >
                {tick.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Metric groups */}
        {metrics.map((metric, mi) => {
          const groupX = PAD.left + mi * groupWidth;
          const barsStart = groupX + (groupWidth - totalBarsW) / 2;

          return (
            <g key={metric}>
              {models.map((model, bi) => {
                const value = model.metrics[metric] ?? 0;
                const barH = (value / Y_MAX) * IH;
                const x = barsStart + bi * (barWidth + barGap);
                const yTop = PAD.top + IH - barH;

                return (
                  <g key={model.short}>
                    <motion.rect
                      x={x}
                      width={barWidth}
                      rx={4}
                      ry={4}
                      fill={model.color}
                      fillOpacity={0.82}
                      initial={{ y: PAD.top + IH, height: 0 }}
                      whileInView={{ y: yTop, height: barH }}
                      viewport={{ once: true, margin: '-48px' }}
                      transition={{
                        duration: 0.75,
                        delay: mi * 0.09 + bi * 0.04,
                        ease: [0.16, 1, 0.3, 1],
                      }}
                    />
                    <motion.text
                      x={x + barWidth / 2}
                      y={yTop - 6}
                      textAnchor="middle"
                      fill="#e2e8f0"
                      fontSize={8.5}
                      fontWeight={500}
                      initial={{ opacity: 0 }}
                      whileInView={{ opacity: 1 }}
                      viewport={{ once: true }}
                      transition={{
                        duration: 0.35,
                        delay: mi * 0.09 + bi * 0.04 + 0.55,
                      }}
                    >
                      {value.toFixed(4)}
                    </motion.text>
                  </g>
                );
              })}
              {/* X-axis metric label */}
              <text
                x={groupX + groupWidth / 2}
                y={PAD.top + IH + 22}
                textAnchor="middle"
                fill="#94a3b8"
                fontSize={11.5}
                fontWeight={600}
              >
                {metric}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 pl-1">
        {models.map((m) => (
          <span key={m.short} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-[3px]"
              style={{ backgroundColor: m.color, opacity: 0.82 }}
            />
            <span className="text-[11px] text-slate-400">{m.short}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* ================================================================
   F1 COMPARISON BARS — horizontal progress-style for Attempt 2
   ================================================================ */

function F1ComparisonBars({
  items,
}: {
  items: { label: string; value: number; color: string }[];
}) {
  const maxVal = Math.max(...items.map((d) => d.value));

  return (
    <div className="mt-4 space-y-3.5">
      {items.map((item, i) => (
        <div key={item.label}>
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
              {item.label}
            </span>
            <span className="font-mono text-sm font-semibold tabular-nums text-slate-100">
              {item.value.toFixed(4)}
            </span>
          </div>
          <div className="relative h-6 w-full overflow-hidden rounded-md bg-slate-900/60">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-md"
              style={{ backgroundColor: item.color, opacity: 0.75 }}
              initial={{ width: '0%' }}
              whileInView={{
                width: `${(item.value / maxVal) * 94}%`,
              }}
              viewport={{ once: true, margin: '-48px' }}
              transition={{
                duration: 0.85,
                delay: i * 0.14,
                ease: [0.16, 1, 0.3, 1],
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

/* ================================================================
   SMALL INFO BADGE — for non-chartable metadata
   ================================================================ */

function InfoBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-green-900/20 bg-slate-900/45 px-3 py-2">
      <dt className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
        {label}
      </dt>
      <dd className="mt-0.5 text-[13px] leading-snug text-slate-200">
        {value}
      </dd>
    </div>
  );
}

/* ================================================================
   MAIN SECTION
   ================================================================ */

export function ResultsAndTuningSection() {
  return (
    <section
      id="results-and-tuning"
      className="border-t border-green-900/30 bg-[radial-gradient(circle_at_top,#142a1f_0%,#102118_52%,#0b1711_100%)] px-4 py-16 text-slate-100 sm:px-6 lg:px-8"
    >
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.22em] text-green-300">
          Performance snapshot
        </p>
        <h2 className="font-heading text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          F1 experiments and what to do next
        </h2>
        <p className="mt-3 max-w-4xl text-sm leading-relaxed text-slate-300 sm:text-base">
          The CNN is currently the safest primary detector, while LCS provides
          rule-level interpretability and is improving under a gated,
          non-regression tuning process.
        </p>

        <div className="mt-8 grid gap-4 lg:grid-cols-2">
          {/* ── Attempt 1 ──────────────────────────────────── */}
          <Card className="border-green-700/45 bg-slate-950/55 text-slate-100 shadow-[0_18px_42px_-30px_rgba(16,185,129,0.34)]">
            <CardContent className="space-y-4 p-5">
              <h3 className="text-lg font-semibold text-white">
                Attempt 1: CNN + LCS imbalance fix and threshold tuning
              </h3>

              {/* Training config (non-numeric, stays as badges) */}
              <div className="rounded-xl border border-green-800/25 bg-slate-950/45 p-4">
                <h4 className="mb-3 text-sm font-semibold tracking-wide text-green-200">
                  CNN training checkpoint
                </h4>
                <dl className="grid gap-2 sm:grid-cols-2">
                  <InfoBadge
                    label="Best validation F1"
                    value="0.5346 (epoch 9)"
                  />
                  <InfoBadge
                    label="Backbone"
                    value="Same CNN backbone + class-weighted loss"
                  />
                  <InfoBadge
                    label="Class weights"
                    value="Normal 1.0, Anomalous 11.09"
                  />
                </dl>
              </div>

              {/* Why F1 is low */}
              <div className="rounded-lg border border-slate-700/40 bg-slate-900/50 px-3.5 py-3 text-[13px] leading-relaxed text-slate-300">
                <span className="font-semibold text-slate-100">Why is F1 low?</span>{' '}
                Anomalous samples are only ~9% of the dataset (class weight 11.09× shows the imbalance). The model hits 90%+ accuracy by correctly classifying the dominant normal class, but struggles to recall the rare anomaly class — low recall drags F1 down. Training also stopped early at epoch 9; longer training with augmentation is the top lever.
              </div>

              {/* Chart — replaces three separate metric grids */}
              <div className="rounded-xl border border-green-800/25 bg-slate-950/45 p-4">
                <h4 className="text-sm font-semibold tracking-wide text-green-200">
                  Test metrics across configurations
                </h4>
                <GroupedBarChart
                  models={attempt1Models}
                  metrics={chartMetrics}
                />
              </div>

              {/* Extra data points that don't fit the chart */}
              <div className="flex flex-wrap gap-2">
                <span className="rounded-md border border-green-900/20 bg-slate-900/45 px-3 py-1.5 text-[11px] text-slate-400">
                  Accuracy: CNN @ 0.50 = 0.9027 · LCS = 0.9125
                </span>
                <span className="rounded-md border border-green-900/20 bg-slate-900/45 px-3 py-1.5 text-[11px] text-slate-400">
                  LCS final population: 2,719 rules
                </span>
                <span className="rounded-md border border-green-900/20 bg-slate-900/45 px-3 py-1.5 text-[11px] text-slate-400">
                  CNN @ 0.67: small F1 gain, mostly via precision
                </span>
              </div>
            </CardContent>
          </Card>

          {/* ── Attempt 2 ──────────────────────────────────── */}
          <Card className="border-green-700/45 bg-slate-950/55 text-slate-100 shadow-[0_18px_42px_-30px_rgba(16,185,129,0.34)]">
            <CardContent className="space-y-4 p-5">
              <h3 className="text-lg font-semibold text-white">
                Attempt 2: Safe LCS sweep with deployment gates
              </h3>

              {/* Sweep setup (non-numeric, stays as badges) */}
              <div className="rounded-xl border border-green-800/25 bg-slate-950/45 p-4">
                <h4 className="mb-3 text-sm font-semibold tracking-wide text-green-200">
                  Safe LCS sweep setup
                </h4>
                <dl className="grid gap-2 sm:grid-cols-2">
                  <InfoBadge
                    label="CNN fixed reference"
                    value="F1 0.4080, AUC 0.7963, threshold 0.67"
                  />
                  <InfoBadge
                    label="Protocol"
                    value="Train/val/test split + validation threshold tuning"
                  />
                  <InfoBadge
                    label="Safety policy"
                    value="Non-regression gates on accuracy, precision, recall, AUC"
                  />
                </dl>
              </div>

              {/* F1 comparison chart */}
              <div className="rounded-xl border border-green-800/25 bg-slate-950/45 p-4">
                <h4 className="text-sm font-semibold tracking-wide text-green-200">
                  F1 comparison — sweep best vs CNN reference
                </h4>
                <F1ComparisonBars items={attempt2Items} />
                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="rounded-md border border-green-900/20 bg-slate-900/45 px-3 py-1.5 text-[11px] text-slate-400">
                    Accuracy / Precision / Recall / AUC: similar to or slightly
                    above original LCS
                  </span>
                  <span className="rounded-md border border-green-900/20 bg-slate-900/45 px-3 py-1.5 text-[11px] text-slate-400">
                    Promotion: not auto-approved (failed at least one strict
                    gate)
                  </span>
                </div>
              </div>

              {/* Callout */}
              <div className="rounded-xl border border-amber-500/35 bg-amber-500/10 p-4 text-sm text-amber-100">
                Best LCS F1 reached 0.4046, close to CNN F1 0.4080, but strict
                gates blocked auto-promotion to avoid silent regressions.
              </div>

              {/* Prioritised next steps */}
              <div className="rounded-xl border border-green-800/25 bg-slate-950/45 p-4">
                <h4 className="mb-3 text-sm font-semibold tracking-wide text-green-200">
                  Highest-impact next steps
                </h4>
                <ul className="space-y-2">
                  {levers.map((l) => (
                    <li
                      key={l.n}
                      className="rounded-md border border-green-900/20 bg-slate-900/45 p-2.5"
                    >
                      <p className="text-xs uppercase tracking-[0.14em] text-slate-400">
                        {l.n}. {l.title}
                      </p>
                      <p className="mt-1 text-sm text-slate-100">{l.text}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}