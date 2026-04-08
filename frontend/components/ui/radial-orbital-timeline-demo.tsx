'use client';

import { Calendar, Clock, Code, FileText, User } from 'lucide-react';

import { RadialOrbitalTimeline, type TimelineItem } from '@/components/ui/radial-orbital-timeline';

const timelineData: TimelineItem[] = [
  {
    id: 1,
    title: 'Planning',
    date: 'Jan 2024',
    content: 'Project planning and requirements gathering phase.',
    category: 'planning',
    icon: Calendar,
    relatedIds: [2],
    status: 'completed',
    impact: 'high',
    evidenceBasis: 'engineering-practice',
  },
  {
    id: 2,
    title: 'Design',
    date: 'Feb 2024',
    content: 'UI and system architecture design for the first working build.',
    category: 'design',
    icon: FileText,
    relatedIds: [1, 3],
    status: 'completed',
    impact: 'high',
    evidenceBasis: 'engineering-practice',
  },
  {
    id: 3,
    title: 'Development',
    date: 'Mar 2024',
    content: 'Core feature implementation and integration work.',
    category: 'development',
    icon: Code,
    relatedIds: [2, 4],
    status: 'in-progress',
    impact: 'medium',
    evidenceBasis: 'benchmark',
  },
  {
    id: 4,
    title: 'Testing',
    date: 'Apr 2024',
    content: 'Validation pass with bug fixes and quality improvements.',
    category: 'testing',
    icon: User,
    relatedIds: [3, 5],
    status: 'pending',
    impact: 'medium',
    evidenceBasis: 'benchmark',
  },
  {
    id: 5,
    title: 'Release',
    date: 'May 2024',
    content: 'Final polishing and release preparation.',
    category: 'release',
    icon: Clock,
    relatedIds: [4],
    status: 'pending',
    impact: 'low',
    evidenceBasis: 'engineering-practice',
  },
];

export function RadialOrbitalTimelineDemo() {
  return <RadialOrbitalTimeline timelineData={timelineData} />;
}
