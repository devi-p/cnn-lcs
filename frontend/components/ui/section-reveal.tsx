'use client';

import { PropsWithChildren } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

import { cn } from '@/lib/utils';

type SectionRevealProps = PropsWithChildren<{
  id?: string;
  className?: string;
  delay?: number;
  as?: 'div' | 'section';
}>;

export function SectionReveal({ children, id, className, delay = 0, as = 'div' }: SectionRevealProps) {
  const prefersReducedMotion = useReducedMotion();
  const MotionTag = as === 'section' ? motion.section : motion.div;

  return (
    <MotionTag
      id={id}
      className={cn(className)}
      initial={prefersReducedMotion ? false : { opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.25, margin: '0px 0px -12% 0px' }}
      transition={
        prefersReducedMotion
          ? { duration: 0 }
          : {
              duration: 0.62,
              delay,
              ease: [0.22, 1, 0.36, 1],
            }
      }
    >
      {children}
    </MotionTag>
  );
}
