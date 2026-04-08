'use client';

import { useEffect, useMemo, useState } from 'react';
import { ArrowRight, BookOpenCheck, Link as LinkIcon, Scale } from 'lucide-react';
import { useReducedMotion } from 'framer-motion';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

export type TimelineItem = {
  id: number;
  title: string;
  date: string;
  content: string;
  category: string;
  icon: React.ElementType;
  relatedIds: number[];
  status: 'completed' | 'in-progress' | 'pending';
  impact: 'high' | 'medium' | 'low';
  evidenceBasis: 'benchmark' | 'literature' | 'engineering-practice';
};

type RadialOrbitalTimelineProps = {
  timelineData: TimelineItem[];
  className?: string;
};

export function RadialOrbitalTimeline({ timelineData, className }: RadialOrbitalTimelineProps) {
  const prefersReducedMotion = useReducedMotion();
  const [expandedId, setExpandedId] = useState<number | null>(timelineData[0]?.id ?? null);
  const [rotationAngle, setRotationAngle] = useState<number>(0);

  useEffect(() => {
    if (prefersReducedMotion) {
      return;
    }

    const timer = setInterval(() => {
      setRotationAngle((prev) => {
        const next = (prev + 0.22) % 360;
        return Number(next.toFixed(3));
      });
    }, 50);

    return () => clearInterval(timer);
  }, [prefersReducedMotion]);

  const relatedSet = useMemo(() => {
    const active = timelineData.find((item) => item.id === expandedId);
    return new Set(active?.relatedIds ?? []);
  }, [expandedId, timelineData]);

  const activeItem = timelineData.find((item) => item.id === expandedId) ?? timelineData[0] ?? null;

  const toggleItem = (id: number) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }

    setExpandedId(id);
  };

  const getStatusStyles = (status: TimelineItem['status']) => {
    if (status === 'completed') {
      return 'text-green-100 bg-green-700/35 border-green-400/45';
    }
    if (status === 'in-progress') {
      return 'text-amber-100 bg-amber-600/30 border-amber-400/45';
    }
    return 'text-slate-200 bg-slate-700/35 border-slate-400/35';
  };

  const getImpactStyles = (impact: TimelineItem['impact']) => {
    if (impact === 'high') {
      return 'text-emerald-100 bg-emerald-700/35 border-emerald-400/45';
    }
    if (impact === 'medium') {
      return 'text-amber-100 bg-amber-700/35 border-amber-400/45';
    }
    return 'text-slate-100 bg-slate-700/40 border-slate-400/35';
  };

  const getImpactLabel = (impact: TimelineItem['impact']) => {
    if (impact === 'high') {
      return 'High';
    }
    if (impact === 'medium') {
      return 'Medium';
    }
    return 'Low';
  };

  const getEvidenceStyles = (basis: TimelineItem['evidenceBasis']) => {
    if (basis === 'benchmark') {
      return 'text-cyan-100 bg-cyan-700/35 border-cyan-400/45';
    }
    if (basis === 'literature') {
      return 'text-violet-100 bg-violet-700/35 border-violet-400/45';
    }
    return 'text-teal-100 bg-teal-700/35 border-teal-400/45';
  };

  const getEvidenceLabel = (basis: TimelineItem['evidenceBasis']) => {
    if (basis === 'engineering-practice') {
      return 'Engineering practice';
    }
    if (basis === 'benchmark') {
      return 'Benchmark';
    }
    return 'Literature';
  };

  const calculateNodePosition = (index: number, total: number) => {
    const angle = ((index / total) * 360 + rotationAngle) % 360;
    const radius = 190;
    const radian = (angle * Math.PI) / 180;

    const x = radius * Math.cos(radian);
    const y = radius * Math.sin(radian);

    const zIndex = Math.round(100 + 50 * Math.cos(radian));
    const opacity = Math.max(0.48, Math.min(1, 0.48 + 0.52 * ((1 + Math.sin(radian)) / 2)));

    return { x, y, zIndex, opacity };
  };

  return (
    <div className={cn('relative w-full overflow-hidden rounded-3xl border border-green-800/35 bg-slate-950/75 p-4 text-slate-100 sm:p-6', className)}>
      <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="relative min-h-107.5">
          <div className="absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full bg-linear-to-br from-green-300 via-emerald-300 to-cyan-300 opacity-30 blur-xl" />
          <div className="absolute left-1/2 top-1/2 h-16 w-16 -translate-x-1/2 -translate-y-1/2 rounded-full border border-green-200/30 bg-slate-900/60" />
          <div className="absolute left-1/2 top-1/2 h-95 w-95 -translate-x-1/2 -translate-y-1/2 rounded-full border border-green-200/15" />

          {timelineData.map((item, index) => {
            const position = calculateNodePosition(index, timelineData.length);
            const Icon = item.icon;
            const isActive = expandedId === item.id;
            const isRelated = relatedSet.has(item.id);

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => toggleItem(item.id)}
                className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-300/70"
                style={{
                  transform: `translate(${position.x}px, ${position.y}px)`,
                  zIndex: isActive ? 250 : position.zIndex,
                  opacity: isActive ? 1 : position.opacity,
                }}
              >
                <div
                  className={cn(
                    'mx-auto flex h-11 w-11 items-center justify-center rounded-full border-2 transition-all duration-300',
                    isActive
                      ? 'scale-125 border-green-200 bg-green-200 text-slate-900 shadow-[0_0_25px_rgba(134,239,172,0.45)]'
                      : isRelated
                      ? 'border-green-300/80 bg-green-100/20 text-green-100'
                      : 'border-green-100/35 bg-slate-900 text-slate-100'
                  )}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div
                  className={cn(
                    'mt-2 whitespace-nowrap text-center text-[0.68rem] font-semibold uppercase tracking-[0.12em] transition-all',
                    isActive ? 'text-green-100' : 'text-slate-300/80'
                  )}
                >
                  {item.title}
                </div>
              </button>
            );
          })}
        </div>

        <Card className="border-green-700/45 bg-slate-950/55 text-slate-100 shadow-[0_24px_44px_-30px_rgba(16,185,129,0.4)]">
          <CardHeader className="pb-3">
            <div className="mt-1 flex items-start justify-between gap-3">
              <CardTitle className="min-w-0 flex-1 text-lg leading-tight text-white">
                {activeItem?.title ?? 'Background overview'}
              </CardTitle>
              <Badge className={cn('shrink-0 border text-[0.65rem] uppercase tracking-[0.12em]', activeItem ? getStatusStyles(activeItem.status) : 'border-slate-600 bg-slate-700 text-slate-100')}>
                {activeItem?.status.replace('-', ' ') ?? 'overview'}
              </Badge>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center rounded-full border border-green-200/20 bg-slate-900/55 px-2.5 py-1 text-[0.66rem] uppercase tracking-[0.12em] text-slate-300">
                {activeItem?.date ?? 'Select a node'}
              </span>

              {activeItem ? (
                <Badge className={cn('border text-[0.66rem] uppercase tracking-[0.12em]', getImpactStyles(activeItem.impact))}>
                  <Scale className="mr-1 h-3.5 w-3.5" />
                  Project impact: {getImpactLabel(activeItem.impact)}
                </Badge>
              ) : null}

              {activeItem ? (
                <Badge className={cn('border text-[0.66rem] tracking-[0.08em]', getEvidenceStyles(activeItem.evidenceBasis))}>
                  <BookOpenCheck className="mr-1 h-3.5 w-3.5" />
                  {getEvidenceLabel(activeItem.evidenceBasis)}
                </Badge>
              ) : null}
            </div>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-slate-300">
            <p>{activeItem?.content ?? 'Select a node to see the key idea and why it matters.'}</p>

            {activeItem ? (
              <>
                {activeItem.relatedIds.length > 0 ? (
                  <div className="mt-3 border-t border-green-200/15 pt-4">
                    <p className="mb-2 inline-flex items-center gap-1 text-xs uppercase tracking-[0.12em] text-slate-300">
                      <LinkIcon className="h-3.5 w-3.5" />
                      Related ideas
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {activeItem.relatedIds.map((relatedId) => {
                        const related = timelineData.find((item) => item.id === relatedId);
                        if (!related) {
                          return null;
                        }

                        return (
                          <Button
                            key={relatedId}
                            variant="outline"
                            size="xs"
                            className="border-green-200/25 bg-transparent text-slate-200 hover:border-green-300/45 hover:bg-green-200/10"
                            onClick={() => toggleItem(relatedId)}
                          >
                            {related.title}
                            <ArrowRight className="h-3 w-3" />
                          </Button>
                        );
                      })}
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}
          </CardContent>
        </Card>
      </div>

    </div>
  );
}
