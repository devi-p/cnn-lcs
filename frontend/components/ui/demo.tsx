'use client';

import { AnomalousMatterHero } from '@/components/ui/anomalous-matter-hero';

export function SplineSceneBasic() {
  return (
    <AnomalousMatterHero
      title="CMPUT 414 Project · CNN-LCS Audio Diagnostics"
      subtitle="Engine anomaly detection for bearings and gearboxes from raw WAV recordings."
      description="Run the deployed FastAPI pipeline, inspect anomaly probability, and review model notes powered by EfficientNet-B0 with optional ExSTraCS support."
    />
  );
}
