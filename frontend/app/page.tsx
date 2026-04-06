import { SplineSceneBasic } from '@/components/ui/demo';
import { UploadAnalyzeCard } from '@/components/upload-analyze-card';
import { SiteHeader } from '@/components/site-header';

const GITHUB_URL = process.env.NEXT_PUBLIC_GITHUB_URL ?? 'https://github.com/devi-p/cnn-lcs';

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,#f8fcf9_0%,#eef5ef_58%)]">
      <SiteHeader githubUrl={GITHUB_URL} />

      <section className="relative min-h-screen">
        <SplineSceneBasic />
      </section>

      <section
        id="upload-section"
        className="blueprint-grid relative flex min-h-screen items-center border-t border-green-800/25 px-4 py-10 sm:px-6 lg:px-8"
      >
        <div className="mx-auto grid w-full max-w-7xl gap-8 md:grid-cols-[1fr_1.1fr] md:items-start">
          <div className="rounded-xl border border-green-800/30 bg-white/85 p-6 text-slate-900 shadow-[0_20px_45px_-32px_rgba(22,101,52,0.34)] md:p-8">
            <p className="mb-3 font-heading text-xs uppercase tracking-[0.22em] text-green-800">Operational Workflow</p>
            <h2 className="mb-5 font-heading text-3xl font-semibold leading-tight text-slate-900 md:text-4xl">
              Upload engine audio and get anomaly confidence in seconds.
            </h2>
            <p className="text-sm leading-relaxed text-slate-700 md:text-base">
              The backend applies the same spectrogram preprocessing and EfficientNet-B0 family used in
              training. If LCS artifacts are present, responses are automatically enriched with combined
              CNN + LCS scoring metadata.
            </p>
          </div>

          <UploadAnalyzeCard />
        </div>
      </section>
    </main>
  );
}
