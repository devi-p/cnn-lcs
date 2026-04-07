"use client";

import { Cpu, Database, ListChecks, SlidersHorizontal, Volume2 } from 'lucide-react';

import { RadialOrbitalTimeline, type TimelineItem } from '@/components/ui/radial-orbital-timeline';

const timelineData: TimelineItem[] = [
  {
    id: 1,
    title: 'Why audio-based detection',
    date: 'Problem',
    content: 'Engine sound can reveal early faults before visible damage, so teams can catch issues sooner.',
    category: 'motivation',
    icon: Volume2,
    relatedIds: [2, 4],
    status: 'completed',
    impact: 'high',
    evidenceBasis: 'engineering-practice',
  },
  {
    id: 2,
    title: 'What benchmarks show',
    date: 'Evidence',
    content: 'DCASE and MIMII benchmarks show that spectrogram plus CNN pipelines can detect machine anomalies reliably.',
    category: 'data',
    icon: Database,
    relatedIds: [1, 3],
    status: 'completed',
    impact: 'high',
    evidenceBasis: 'benchmark',
  },
  {
    id: 3,
    title: 'Why learned features',
    date: 'Approach',
    content: 'CNN embeddings on log-mel spectrograms reduce manual feature tuning and generalize better across conditions.',
    category: 'modeling',
    icon: SlidersHorizontal,
    relatedIds: [2, 4],
    status: 'completed',
    impact: 'medium',
    evidenceBasis: 'literature',
  },
  {
    id: 4,
    title: 'Why explainable rules',
    date: 'Interpretability',
    content: 'ExSTraCS-style IF-THEN rules show why a sample is flagged, improving trust compared with black-box outputs.',
    category: 'interpretability',
    icon: ListChecks,
    relatedIds: [1, 3],
    status: 'completed',
    impact: 'high',
    evidenceBasis: 'literature',
  },
];

export function BackgroundResearchSection() {
  return (
    <section
      id="background-research-section"
      className="border-y border-green-900/30 bg-[radial-gradient(circle_at_top,#0f2419_0%,#0c1a13_52%,#0a1610_100%)] px-4 py-16 text-slate-100 sm:px-6 lg:px-8"
    >
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.24em] text-green-300">Why this design</p>
        <h2 className="font-heading text-3xl font-semibold tracking-tight text-white sm:text-4xl">Background and motivation</h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-300 sm:text-base">
          These four ideas explain why we built the CNN plus LCS system this way.
        </p>

        <div className="mt-8">
          <RadialOrbitalTimeline timelineData={timelineData} />
        </div>
      </div>
    </section>
  );
}
