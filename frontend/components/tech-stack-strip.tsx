import { Database, Globe, Server, Sigma } from 'lucide-react';

type StackItem = {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  bullets: string[];
  secondaryIcon?: React.ComponentType<{ className?: string }>;
};

const stack: StackItem[] = [
  {
    title: 'Data & preprocessing',
    icon: Database,
    bullets: [
      'Engine audio (bearing & gearbox)',
      'Python + librosa for segmentation and spectrograms',
      'Segment-level slicing creates consistent training windows',
      'Dataset split pipeline keeps train and test leakage-safe',
    ],
  },
  {
    title: 'Models',
    icon: Sigma,
    bullets: [
      'EfficientNet-B0 CNN feature extractor',
      'ExSTraCS Learning Classifier System',
      'Hybrid output combines confidence and rule-level explanation',
      'Feature-to-rule flow preserves interpretability after deep encoding',
    ],
  },
  {
    title: 'App & serving',
    icon: Server,
    bullets: [
      'FastAPI inference backend',
      'Next.js + Tailwind + shadcn UI + Spline 3D',
      'Frontend and backend communicate through typed JSON responses',
      'Outputs include optional LCS metadata for explainable diagnostics',
    ],
    secondaryIcon: Globe,
  },
];

export function TechStackStrip() {
  return (
    <section id="tech-stack-section" className="border-t border-green-800/20 px-4 py-14 sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-7xl">
        <p className="mb-2 font-heading text-xs uppercase tracking-[0.22em] text-green-800">Build Stack</p>
        <h2 className="font-heading text-2xl font-semibold text-slate-900 sm:text-3xl">Tech stack</h2>
        <p className="mt-2 mb-7 max-w-3xl text-sm leading-relaxed text-slate-600 sm:text-base">
          End-to-end components used for preprocessing, model learning, and serving.
        </p>

        <div className="space-y-4">
          {stack.map((item) => (
            <div
              key={item.title}
              className="grid items-start gap-4 rounded-xl border border-green-800/30 bg-white/90 p-5 shadow-[0_12px_30px_-24px_rgba(22,101,52,0.25)] transition-transform duration-200 hover:-translate-y-0.5 sm:grid-cols-[200px_1fr]"
            >
              <div className="flex items-center gap-3">
                <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-green-100 text-green-800">
                  <item.icon className="h-4 w-4" />
                </span>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-semibold text-slate-900">{item.title}</h3>
                  {item.secondaryIcon ? <item.secondaryIcon className="h-4 w-4 text-green-700" /> : null}
                </div>
              </div>

              <ul className="space-y-1.5 text-sm text-slate-600">
                {item.bullets.map((bullet) => (
                  <li key={bullet} className="flex gap-2">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-green-700" />
                    <span>{bullet}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
