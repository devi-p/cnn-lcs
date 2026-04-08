# Frontend (Next.js + TypeScript)

This frontend provides the demo UI for CNN-LCS audio anomaly inference.

## Stack

- Next.js (App Router) + TypeScript
- Tailwind CSS
- shadcn UI configuration
- Spline runtime for 3D hero rendering
- Three.js for anomalous matter hero rendering
- Framer Motion + Lucide icons

## Why `/components/ui`

The repository originally had no frontend. This app was scaffolded and configured with shadcn so reusable primitives live in `components/ui` and can be imported consistently as `@/components/ui/*`.

`cn` helper is available at `@/lib/utils`.

## Install and Run

From the `frontend` directory:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Backend API

The upload flow posts to:

- `POST /api/analyze-audio`

Set backend base URL with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

If unset, the app defaults to `http://127.0.0.1:8000`.

## Main UI Files

- `app/page.tsx` - landing page composition
- `components/ui/demo.tsx` - hero entry wrapper
- `components/ui/anomalous-matter-hero.tsx` - generative Three.js hero scene
- `components/ui/splite.tsx` - lazy Spline scene wrapper
- `components/ui/spotlight.tsx` - decorative spotlight effect
- `components/upload-analyze-card.tsx` - upload/analyze/result flow

## Spline Scene

The hero uses an industrial-style Spline scene URL in `components/ui/demo.tsx`.
You can swap the 3D model without code changes by setting:

```bash
NEXT_PUBLIC_SPLINE_SCENE_URL=https://prod.spline.design/your-engine-scene/scene.splinecode
```

If this variable is not set, the default scene in `components/ui/demo.tsx` is used.

The Spline wrapper supports both URL types:

- `my.spline.design/...` share links: rendered in an iframe mode (recommended for shared scenes)
- `...scene.splinecode` links: rendered in runtime mode (`@splinetool/react-spline`)

For an engine-style replacement, search Spline Community for terms like:

- `engine`
- `gear`
- `piston`
- `mechanical blueprint`
