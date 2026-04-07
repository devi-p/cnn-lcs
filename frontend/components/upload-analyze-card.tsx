'use client';

import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { Activity, AlertTriangle, AudioLines, Loader2, UploadCloud, X } from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

type AnalyzeResponse = {
  status: string;
  anomaly_probability: number;
  label: 'Normal' | 'Anomalous';
  machine_type: string;
  notes: string;
  segments_analyzed?: number;
  threshold?: number;
  inference_source?: string;
  cnn_probability?: number;
  lcs_probability?: number;
  lcs_run_id?: string | null;
  lcs_reasoning_note?: string;
  lcs_aliases_available?: boolean;
  lcs_reasons?: Array<{
    rule_id: number;
    then_prediction: string;
    match_rate: number;
    matched_segments: number;
    total_segments: number;
    confidence: number;
    support: number;
    condition_text: string;
  }>;
};

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? '').replace(/\/$/, '');
const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024;
const MAX_DURATION_SECONDS = 15;

async function getAudioDurationSeconds(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const audio = document.createElement('audio');
    const objectUrl = URL.createObjectURL(file);

    audio.preload = 'metadata';
    audio.src = objectUrl;

    audio.onloadedmetadata = () => {
      const duration = Number.isFinite(audio.duration) ? audio.duration : NaN;
      URL.revokeObjectURL(objectUrl);
      if (Number.isNaN(duration)) {
        reject(new Error('Could not read audio duration.'));
        return;
      }
      resolve(duration);
    };

    audio.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error('Could not read audio metadata.'));
    };
  });
}

export function UploadAnalyzeCard() {
  const prefersReducedMotion = useReducedMotion();
  const [file, setFile] = useState<File | null>(null);
  const [machineType, setMachineType] = useState<'bearing' | 'gearbox'>('bearing');
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!isLoading) {
      setElapsedSeconds(0);
      return;
    }
    const id = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [isLoading]);

  const cancelAnalysis = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
    setError('Analysis cancelled.');
  }, []);

  const probabilityPercent = useMemo(() => {
    if (!result) {
      return 0;
    }
    return Math.round(result.anomaly_probability * 100);
  }, [result]);

  const handleFileSelect = async (candidate: File | null) => {
    if (!candidate) {
      return;
    }

    if (!candidate.name.toLowerCase().endsWith('.wav')) {
      setError('Please select a WAV file (.wav).');
      return;
    }

    if (candidate.size > MAX_FILE_SIZE_BYTES) {
      setError('File is too large. Please use a WAV file under 25MB.');
      return;
    }

    try {
      const duration = await getAudioDurationSeconds(candidate);
      if (duration > MAX_DURATION_SECONDS) {
        setError('Recording is too long. Please upload a clip shorter than 15 seconds.');
        return;
      }
    } catch {
      setError('Unable to read audio metadata. Please try another WAV file.');
      return;
    }

    setError(null);
    setResult(null);
    setFile(candidate);
  };

  const onInputChange = async (event: ChangeEvent<HTMLInputElement>) => {
    await handleFileSelect(event.target.files?.[0] ?? null);
  };

  const onDrop = async (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);
    await handleFileSelect(event.dataTransfer.files?.[0] ?? null);
  };

  const analyzeFile = async () => {
    if (!file) {
      setError('Attach a WAV file before analysis.');
      return;
    }

    setError(null);
    setIsLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('machine_type', machineType);

      const response = await fetch(`${API_BASE}/api/analyze-audio`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? 'Backend error while analyzing audio.');
      }

      setResult(payload as AnalyzeResponse);
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === 'AbortError') {
        return;
      }
      let message: string;
      if (requestError instanceof TypeError && requestError.message === 'Failed to fetch') {
        message = 'Unable to reach the analysis server. It may be starting up — please try again in a moment.';
      } else {
        message = requestError instanceof Error ? requestError.message : 'Unexpected request error.';
      }
      setError(message);
    } finally {
      abortControllerRef.current = null;
      setIsLoading(false);
    }
  };

  return (
    <Card className="border-green-800/30 bg-white/92 text-slate-900 shadow-[0_22px_60px_-34px_rgba(22,101,52,0.38)] transition-[transform,box-shadow] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] hover:-translate-y-0.5 hover:shadow-[0_24px_70px_-34px_rgba(22,101,52,0.44)]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-2xl text-slate-900">
          <AudioLines className="h-5 w-5 text-green-800" />
          Upload and Analyze
        </CardTitle>
        <CardDescription className="text-slate-600">
          Drop a 5-15 second engine recording and run CNN-LCS inference.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        <label
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          className={`group flex min-h-36 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-6 text-center transition-[border-color,background-color,color,transform,box-shadow] duration-300 ease-[cubic-bezier(0.25,1,0.5,1)] ${
            isDragging
              ? 'border-green-700 bg-green-100 text-green-900 shadow-[0_10px_35px_-24px_rgba(22,101,52,0.6)]'
              : 'border-slate-300/70 bg-white text-slate-600 hover:border-green-700 hover:text-slate-900'
          }`}
        >
          <UploadCloud className="h-7 w-7" />
          <div>
            <p className="text-sm font-medium">
              {file ? file.name : 'Drag and drop a .wav file, or click to browse'}
            </p>
            <p className="text-xs opacity-80">WAV, under 25MB, 5-15s recommended (16kHz mono or stereo)</p>
          </div>
          <input className="hidden" type="file" accept=".wav,audio/wav" onChange={onInputChange} />
        </label>

        <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-[0.2em] text-slate-500">Machine Type</label>
            <select
              value={machineType}
              onChange={(event) => setMachineType(event.target.value as 'bearing' | 'gearbox')}
              className="w-full rounded-lg border border-green-800/35 bg-white px-3 py-2.5 text-sm text-slate-900 outline-none ring-green-700 transition focus:ring-2"
            >
              <option value="bearing">Bearing</option>
              <option value="gearbox">Gearbox</option>
            </select>
          </div>

          <div className="flex items-end gap-2">
            <motion.button
              type="button"
              onClick={analyzeFile}
              disabled={isLoading || !file}
              whileTap={prefersReducedMotion ? undefined : { scale: 0.98 }}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-green-700 px-5 text-sm font-semibold text-white transition hover:bg-green-600 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing{elapsedSeconds > 0 ? `... ${elapsedSeconds}s` : '...'}
                </>
              ) : (
                <>
                  <Activity className="h-4 w-4" />
                  Analyze
                </>
              )}
            </motion.button>

            {isLoading && (
              <button
                type="button"
                onClick={cancelAnalysis}
                className="inline-flex h-11 items-center justify-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 text-sm font-medium text-slate-700 transition hover:bg-slate-100"
              >
                <X className="h-3.5 w-3.5" />
                Cancel
              </button>
            )}
          </div>
        </div>

        <AnimatePresence initial={false}>
          {error ? (
            <motion.div
              key="analysis-error"
              initial={prefersReducedMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={prefersReducedMotion ? { opacity: 0 } : { opacity: 0, y: -6 }}
              transition={
                prefersReducedMotion
                  ? { duration: 0 }
                  : {
                      duration: 0.2,
                      ease: [0.25, 1, 0.5, 1],
                    }
              }
              className="rounded-lg border border-amber-400/35 bg-amber-50 p-3 text-sm text-amber-900"
            >
              <div className="mb-1 inline-flex items-center gap-2 font-medium">
                <AlertTriangle className="h-4 w-4" />
                Analysis failed
              </div>
              <p>{error}</p>
              <p className="mt-1 text-xs text-amber-800/90">Try another WAV clip or refresh the page if this keeps happening.</p>
            </motion.div>
          ) : null}
        </AnimatePresence>

        <AnimatePresence initial={false}>
          {result ? (
            <motion.div
              key={`analysis-result-${result.label}-${probabilityPercent}`}
              initial={prefersReducedMotion ? false : { opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={prefersReducedMotion ? { opacity: 0 } : { opacity: 0, y: -10 }}
              transition={
                prefersReducedMotion
                  ? { duration: 0 }
                  : {
                      duration: 0.34,
                      ease: [0.22, 1, 0.36, 1],
                    }
              }
              className="space-y-4 rounded-xl border border-green-800/30 bg-white p-4"
            >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Result</p>
                <p className="text-lg font-semibold text-slate-900">{result.label}</p>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider ${
                  result.label === 'Anomalous'
                    ? 'bg-slate-200 text-slate-900'
                    : 'bg-green-200 text-green-900'
                }`}
              >
                {result.label}
              </span>
            </div>

            <div>
              <div className="mb-1 flex items-center justify-between text-sm text-slate-700">
                <span>Anomaly probability</span>
                <span className="font-semibold text-slate-900">{probabilityPercent}%</span>
              </div>
              <div className="h-2 rounded-full bg-slate-200">
                <div
                  className="h-2 rounded-full bg-linear-to-r from-green-800 via-green-600 to-green-700 transition-[width] duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]"
                  style={{ width: `${probabilityPercent}%` }}
                />
              </div>
            </div>

            <div className="grid gap-2 text-sm text-slate-700 sm:grid-cols-2">
              <p>
                <span className="text-slate-500">Machine:</span> {result.machine_type}
              </p>
              <p>
                <span className="text-slate-500">Source:</span> {result.inference_source ?? 'cnn'}
              </p>
              <p>
                <span className="text-slate-500">Segments:</span> {result.segments_analyzed ?? '-'}
              </p>
              <p>
                <span className="text-slate-500">Threshold:</span>{' '}
                {typeof result.threshold === 'number' ? result.threshold.toFixed(2) : '-'}
              </p>
              <p>
                <span className="text-slate-500">CNN score:</span>{' '}
                {typeof result.cnn_probability === 'number' ? result.cnn_probability.toFixed(4) : '-'}
              </p>
              <p>
                <span className="text-slate-500">LCS score:</span>{' '}
                {typeof result.lcs_probability === 'number' ? result.lcs_probability.toFixed(4) : 'not active'}
              </p>
              {result.lcs_run_id ? (
                <p className="sm:col-span-2">
                  <span className="text-slate-500">LCS run:</span> {result.lcs_run_id}
                </p>
              ) : null}
            </div>

            <p className="rounded-md border border-green-800/25 bg-green-50 p-3 text-sm text-slate-800">
              {result.notes}
            </p>

            {result.lcs_reasons && result.lcs_reasons.length > 0 ? (
              <details className="rounded-md border border-green-800/25 bg-white p-3 transition-colors duration-200 ease-[cubic-bezier(0.25,1,0.5,1)]">
                <summary className="cursor-pointer text-sm font-semibold text-slate-900">
                  Why this LCS decision? (matched IF-THEN rules)
                </summary>

                <p className="mt-2 text-xs text-slate-600">
                  {result.lcs_reasoning_note ??
                    'Rule evidence is based on learned latent features. Any named aliases are approximate and not definitive causal proof.'}
                </p>

                <p className="mt-1 text-xs text-slate-500">
                  Glossary: CNN = anomaly scoring model, LCS = IF-THEN rule layer.
                </p>

                <p className="mt-1 text-xs text-slate-500">
                  Alias map loaded: {result.lcs_aliases_available ? 'yes' : 'no'}
                </p>

                <div className="mt-3 space-y-2">
                  {result.lcs_reasons.map((reason) => (
                    <div key={reason.rule_id} className="rounded-md border border-green-900/20 bg-green-50/60 p-3">
                      <div className="mb-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-700">
                        <span>
                          <span className="text-slate-500">Rule:</span> #{reason.rule_id}
                        </span>
                        <span>
                          <span className="text-slate-500">THEN:</span> {reason.then_prediction}
                        </span>
                        <span>
                          <span className="text-slate-500">Match:</span>{' '}
                          {(reason.match_rate * 100).toFixed(1)}% ({reason.matched_segments}/{reason.total_segments} segments)
                        </span>
                        <span>
                          <span className="text-slate-500">Confidence:</span> {(reason.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-xs leading-relaxed text-slate-800">{reason.condition_text}</p>
                    </div>
                  ))}
                </div>
              </details>
            ) : null}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
