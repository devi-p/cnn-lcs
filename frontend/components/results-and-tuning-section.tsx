import { Card, CardContent } from '@/components/ui/card';

type Metric = {
  label: string;
  value: string;
};

type MetricBlock = {
  title: string;
  metrics: Metric[];
};

const attemptOneBlocks: MetricBlock[] = [
  {
    title: 'CNN training checkpoint',
    metrics: [
      { label: 'Best validation F1', value: '0.5346 (epoch 9)' },
      { label: 'Backbone', value: 'Same CNN backbone + class-weighted loss' },
      { label: 'Class weights', value: 'Normal 1.0, Anomalous 11.09' },
    ],
  },
  {
    title: 'CNN test at threshold 0.50',
    metrics: [
      { label: 'Accuracy', value: '0.9027' },
      { label: 'F1', value: '0.4036' },
      { label: 'Precision', value: '0.4125' },
      { label: 'Recall', value: '0.3950' },
      { label: 'AUC', value: '0.7963' },
    ],
  },
  {
    title: 'CNN test at tuned threshold 0.67',
    metrics: [
      { label: 'F1', value: '0.4080' },
      { label: 'Precision', value: '0.4751' },
      { label: 'Recall', value: '0.3575' },
      { label: 'Delta vs threshold 0.50', value: 'Small F1 gain, mostly via precision' },
    ],
  },
  {
    title: 'LCS on CNN features (k = 50 / 1280)',
    metrics: [
      { label: 'F1', value: '0.3805' },
      { label: 'Accuracy', value: '0.9125' },
      { label: 'Precision', value: '0.4640' },
      { label: 'Recall', value: '0.3225' },
      { label: 'AUC', value: '0.7347' },
      { label: 'Final population', value: '2,719 rules' },
    ],
  },
];

const attemptTwoBlocks: MetricBlock[] = [
  {
    title: 'Safe LCS sweep setup',
    metrics: [
      { label: 'CNN fixed reference', value: 'F1 0.4080, AUC 0.7963, threshold 0.67' },
      { label: 'Protocol', value: 'Train/val/test split + validation threshold tuning' },
      { label: 'Safety policy', value: 'Non-regression gates on accuracy, precision, recall, AUC' },
    ],
  },
  {
    title: 'Best LCS run from sweep',
    metrics: [
      { label: 'F1', value: '0.4046 (up from 0.3805)' },
      { label: 'Accuracy/Precision/Recall/AUC', value: 'Similar to or slightly above original LCS' },
      { label: 'Promotion status', value: 'Not auto-approved (failed at least one strict gate)' },
    ],
  },
];

const prioritizedLevers: Metric[] = [
  {
    label: '1. Better CNN training (highest impact)',
    value: 'Train to ~30 epochs and add SpecAugment so LCS gets stronger features.',
  },
  {
    label: '2. Threshold search as a first-class step',
    value: 'Keep validating threshold policy; anomaly-heavy recall settings can still improve F1 trade-offs.',
  },
  {
    label: '3. Longer LCS search budget',
    value: 'Increase ExSTraCS iterations to 100,000 with N = 3000 for rare anomaly rules.',
  },
  {
    label: '4. LCS representation capacity',
    value: 'Raise selected features from k = 50 toward k = 100-150 and retune mutation/hyperparameters.',
  },
];

function MetricGrid({ block }: { block: MetricBlock }) {
  return (
    <div className="rounded-xl border border-green-800/25 bg-slate-950/45 p-4">
      <h4 className="mb-3 text-sm font-semibold tracking-wide text-green-200">{block.title}</h4>
      <dl className="grid gap-2 sm:grid-cols-2">
        {block.metrics.map((metric) => (
          <div key={`${block.title}-${metric.label}`} className="rounded-md border border-green-900/20 bg-slate-900/45 p-2.5">
            <dt className="text-[11px] uppercase tracking-[0.16em] text-slate-400">{metric.label}</dt>
            <dd className="mt-1 text-sm text-slate-100">{metric.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function ResultsAndTuningSection() {
  return (
    <section
      id="results-and-tuning"
      className="border-t border-green-900/30 bg-[radial-gradient(circle_at_top,#142a1f_0%,#102118_52%,#0b1711_100%)] px-4 py-16 text-slate-100 sm:px-6 lg:px-8"
    >
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.22em] text-green-300">Performance snapshot</p>
        <h2 className="font-heading text-3xl font-semibold tracking-tight text-white sm:text-4xl">F1 experiments and what to do next</h2>
        <p className="mt-3 max-w-4xl text-sm leading-relaxed text-slate-300 sm:text-base">
          The CNN is currently the safest primary detector, while LCS provides rule-level interpretability and is improving under a gated,
          non-regression tuning process.
        </p>

        <div className="mt-8 grid gap-4 lg:grid-cols-2">
          <Card className="border-green-700/45 bg-slate-950/55 text-slate-100 shadow-[0_18px_42px_-30px_rgba(16,185,129,0.34)]">
            <CardContent className="space-y-4 p-5">
              <h3 className="text-lg font-semibold text-white">Attempt 1: CNN + LCS imbalance fix and threshold tuning</h3>
              {attemptOneBlocks.map((block) => (
                <MetricGrid key={block.title} block={block} />
              ))}
            </CardContent>
          </Card>

          <Card className="border-green-700/45 bg-slate-950/55 text-slate-100 shadow-[0_18px_42px_-30px_rgba(16,185,129,0.34)]">
            <CardContent className="space-y-4 p-5">
              <h3 className="text-lg font-semibold text-white">Attempt 2: Safe LCS sweep with deployment gates</h3>
              {attemptTwoBlocks.map((block) => (
                <MetricGrid key={block.title} block={block} />
              ))}

              <div className="rounded-xl border border-amber-500/35 bg-amber-500/10 p-4 text-sm text-amber-100">
                Best LCS F1 reached 0.4046, close to CNN F1 0.4080, but strict gates blocked auto-promotion to avoid silent regressions.
              </div>

              <div className="rounded-xl border border-green-800/25 bg-slate-950/45 p-4">
                <h4 className="mb-3 text-sm font-semibold tracking-wide text-green-200">Highest-impact next steps</h4>
                <ul className="space-y-2">
                  {prioritizedLevers.map((lever) => (
                    <li key={lever.label} className="rounded-md border border-green-900/20 bg-slate-900/45 p-2.5">
                      <p className="text-xs uppercase tracking-[0.14em] text-slate-400">{lever.label}</p>
                      <p className="mt-1 text-sm text-slate-100">{lever.value}</p>
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