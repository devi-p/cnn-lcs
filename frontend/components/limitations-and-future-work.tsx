import { AlertCircle, Cpu, Database, ListChecks } from 'lucide-react';

import { Card, CardContent } from '@/components/ui/card';

type LimitationCard = {
  title: string;
  Icon: React.ComponentType<{ className?: string }>;
  bullets: string[];
};

const limitationCards: LimitationCard[] = [
  {
    title: 'Limited datasets',
    Icon: Database,
    bullets: [
      'Current runs focus on bearing and gearbox machine sounds.',
      'Original scope was broader engine anomalies across more components.',
      'Benchmark-style data may not represent full road and hardware variability.',
      'Future: add fans, valves, and real in-car recordings.',
    ],
  },
  {
    title: 'Image CNN, not audio-native',
    Icon: Cpu,
    bullets: [
      'EfficientNet-B0 is ImageNet-pretrained, then adapted to log-mel inputs.',
      'This choice keeps implementation simple and training more efficient.',
      'Audio-native encoders like PANNs, AST, and YAMNet are strong baselines.',
      'Future: benchmark audio-native models against current EfficientNet.',
    ],
  },
  {
    title: 'Rule clarity vs raw F1',
    Icon: ListChecks,
    bullets: [
      'CNN currently provides the primary anomaly decision signal.',
      'LCS gives interpretable IF-THEN rules but trails slightly in F1.',
      'Current runs are around low-0.4 CNN F1 vs high-0.3 LCS F1.',
      'Future: deeper LCS tuning and hybrid decision calibration.',
    ],
  },
  {
    title: 'From lab demo to real engines',
    Icon: AlertCircle,
    bullets: [
      'Benchmarks are controlled compared with real driving conditions.',
      'Mic type and placement shifts can reduce model robustness.',
      'Cross-device and in-car validation is still pending.',
      'Future: test robustness across microphones, vehicles, and routes.',
    ],
  },
];

export function LimitationsAndFutureWork() {
  return (
    <section
      id="limitations-section"
      className="border-t border-green-900/25 bg-[radial-gradient(circle_at_top,#0e2318_0%,#0b1913_56%,#08130f_100%)] px-4 py-16 text-slate-100 sm:px-6 lg:px-8"
    >
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.22em] text-green-300">Prototype reality check</p>
        <h2 className="font-heading text-3xl font-semibold tracking-tight text-white sm:text-4xl">Limitations & future directions</h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-300 sm:text-base">
          What we struggled with in this prototype, and how we would improve it next.
        </p>

        <div className="mt-8 grid gap-4 md:grid-cols-2">
          {limitationCards.map((card) => (
            <div key={card.title}>
              <Card
                className="h-full border-green-700/45 bg-slate-950/55 text-slate-100 shadow-[0_18px_42px_-30px_rgba(16,185,129,0.34)] transition-transform duration-200 hover:-translate-y-0.5"
              >
                <CardContent className="p-5">
                  <div className="mb-3 flex items-center gap-3">
                    <span className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-green-500/15 text-green-300">
                      <card.Icon className="h-4 w-4" />
                    </span>
                    <h3 className="text-base font-semibold text-white">{card.title}</h3>
                  </div>

                  <ul className="space-y-2">
                    {card.bullets.map((bullet) => (
                      <li key={bullet} className="flex gap-2 text-sm leading-relaxed text-slate-300">
                        <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-green-300/90" />
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
