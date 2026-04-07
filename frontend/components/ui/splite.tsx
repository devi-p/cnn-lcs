'use client';

import React, { Suspense, lazy } from 'react';

const Spline = lazy(() => import('@splinetool/react-spline'));

interface SplineSceneProps {
  scene: string;
  className?: string;
}

type SceneResolution =
  | { mode: 'runtime'; url: string }
  | { mode: 'iframe'; url: string }
  | { mode: 'invalid'; url: string; reason: string };

function resolveSplineScene(rawSceneUrl: string): SceneResolution {
  try {
    const url = new URL(rawSceneUrl);
    const cleanPath = url.pathname.replace(/^\/+|\/+$/g, '');

    if (url.hostname === 'my.spline.design') {
      return { mode: 'iframe', url: rawSceneUrl };
    }

    if (cleanPath.endsWith('.splinecode')) {
      return { mode: 'runtime', url: url.toString() };
    }

    if (url.hostname === 'prod.spline.design') {
      return { mode: 'runtime', url: `https://prod.spline.design/${cleanPath}/scene.splinecode` };
    }

    return {
      mode: 'invalid',
      url: rawSceneUrl,
      reason: 'Only my.spline.design share links or .splinecode runtime links are supported.',
    };
  } catch {
    return { mode: 'invalid', url: rawSceneUrl, reason: 'The provided URL is not valid.' };
  }
}

class SplineErrorBoundary extends React.Component<
  { children: React.ReactNode; className?: string },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode; className?: string }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className={`flex h-full w-full items-center justify-center p-6 text-center text-slate-300 ${this.props.className ?? ''}`}>
          <p>Unable to render this Spline scene in runtime mode. Use a my.spline.design share link or a valid .splinecode URL.</p>
        </div>
      );
    }
    return this.props.children;
  }
}

export function SplineScene({ scene, className }: SplineSceneProps) {
  const resolved = resolveSplineScene(scene);

  if (resolved.mode === 'invalid') {
    return (
      <div className={`flex h-full w-full items-center justify-center p-6 text-center text-slate-300 ${className ?? ''}`}>
        <p>{resolved.reason}</p>
      </div>
    );
  }

  if (resolved.mode === 'iframe') {
    return (
      <iframe
        src={resolved.url}
        className={className}
        title="Spline Scene"
        allow="fullscreen; xr-spatial-tracking"
        loading="lazy"
      />
    );
  }

  return (
    <SplineErrorBoundary className={className}>
      <Suspense
        fallback={
          <div className="flex h-full w-full items-center justify-center">
            <span className="loader" aria-label="Loading 3D scene" />
          </div>
        }
      >
        <Spline scene={resolved.url} className={className} />
      </Suspense>
    </SplineErrorBoundary>
  );
}
