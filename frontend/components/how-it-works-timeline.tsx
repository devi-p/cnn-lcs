import { Activity, AudioWaveform, ChevronRight, Cpu, ListChecks } from 'lucide-react';

const steps = [
  {
    title: 'Engine sound in',
    caption: 'Short bearing/gearbox recordings.',
    Icon: AudioWaveform,
  },
  {
    title: 'Spectrogram view',
    caption: 'Convert audio into 128-bin log-mel spectrograms.',
    Icon: Activity,
  },
  {
    title: 'CNN features',
    caption: 'EfficientNet-B0 learns a 1280-d feature vector.',
    Icon: Cpu,
  },
  {
    title: 'LCS rules',
    caption: 'ExSTraCS turns features into IF-THEN rules.',
    Icon: ListChecks,
  },
];

export function HowItWorksTimeline() {
  return (
    <section id="how-it-works-section" className="border-t border-green-800/20 px-4 py-14 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.22em] text-green-800">System Flow</p>
        <h2 className="mb-7 font-heading text-2xl font-semibold text-slate-900 sm:text-3xl">How it works</h2>

        <ol className="flex flex-col gap-6 md:flex-row md:items-start md:gap-0">
          {steps.map((step, index) => (
            <li key={step.title} className="flex flex-1 items-start gap-4 md:flex-col md:items-center md:text-center">
              <div className="flex items-center gap-3 md:flex-col md:gap-2">
                <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-green-100 text-green-800 ring-2 ring-green-200/60">
                  <step.Icon className="h-4 w-4" />
                </span>
                <span className="font-heading text-xs font-semibold uppercase tracking-wide text-green-700 md:mt-1">
                  Step {index + 1}
                </span>
              </div>

              <div className="md:mt-2">
                <h3 className="text-sm font-semibold text-slate-900">{step.title}</h3>
                <p className="mt-0.5 text-sm leading-relaxed text-slate-600">{step.caption}</p>
              </div>

              {index < steps.length - 1 && (
                <ChevronRight className="mt-1 hidden h-5 w-5 shrink-0 text-green-600/60 md:mt-5 md:block" />
              )}
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
