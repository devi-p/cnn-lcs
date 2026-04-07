import { BookOpenText, Factory, Microscope } from 'lucide-react';

import { Card, CardContent } from '@/components/ui/card';

const researchItems = [
  {
    title: 'DCASE challenges',
    description: 'Deep CNN approaches for anomalous industrial sounds.',
    Icon: BookOpenText,
  },
  {
    title: 'MIMII dataset',
    description: 'Benchmark recordings of malfunctioning industrial machines.',
    Icon: Factory,
  },
  {
    title: 'ExSTraCS LCS',
    description: 'Rule-based engine fault detection with explainable rules.',
    Icon: Microscope,
  },
];

export function BackgroundResearchStrip() {
  return (
    <section className="border-y border-green-800/20 px-4 py-14 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.22em] text-green-800">
          Background research (very short)
        </p>
        <h2 className="mb-7 font-heading text-2xl font-semibold text-slate-900 sm:text-3xl">
          Background research (very short)
        </h2>

        <div className="grid gap-3 md:grid-cols-3">
          {researchItems.map((item) => (
            <Card
              key={item.title}
              className="border-green-800/30 bg-white/90 text-slate-900 shadow-[0_16px_34px_-30px_rgba(22,101,52,0.3)]"
            >
              <CardContent className="flex items-start gap-3 p-4">
                <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-green-100 text-green-800">
                  <item.Icon className="h-4 w-4" />
                </span>
                <div>
                  <p className="text-sm font-semibold text-slate-900">{item.title}</p>
                  <p className="mt-1 text-sm text-slate-600">{item.description}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
