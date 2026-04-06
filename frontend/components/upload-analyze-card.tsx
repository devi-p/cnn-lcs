'use client';

import { ChangeEvent, DragEvent, useMemo, useState } from 'react';
import { Activity, AlertTriangle, AudioLines, Loader2, UploadCloud } from 'lucide-react';

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
};

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

export function UploadAnalyzeCard() {
  const [file, setFile] = useState<File | null>(null);
  const [machineType, setMachineType] = useState<'bearing' | 'gearbox'>('bearing');
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const probabilityPercent = useMemo(() => {
    if (!result) {
      return 0;
    }
    return Math.round(result.anomaly_probability * 100);
  }, [result]);

  const handleFileSelect = (candidate: File | null) => {
    if (!candidate) {
      return;
    }
    if (!candidate.name.toLowerCase().endsWith('.wav')) {
      setError('Please select a .wav file.');
      return;
    }

    setError(null);
    setResult(null);
    setFile(candidate);
  };

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(event.target.files?.[0] ?? null);
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);
    handleFileSelect(event.dataTransfer.files?.[0] ?? null);
  };

  const analyzeFile = async () => {
    if (!file) {
      setError('Attach a WAV file before analysis.');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('machine_type', machineType);

      const response = await fetch(`${API_BASE}/api/analyze-audio`, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? 'Backend error while analyzing audio.');
      }

      setResult(payload as AnalyzeResponse);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : 'Unexpected request error.';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="border-green-800/30 bg-white/92 text-slate-900 shadow-[0_22px_60px_-34px_rgba(22,101,52,0.38)]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-2xl text-slate-900">
          <AudioLines className="h-5 w-5 text-green-800" />
          Upload and Analyze
        </CardTitle>
        <CardDescription className="text-slate-600">
          Drop a ~10 second engine recording and run real CNN-LCS backend inference.
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
          className={`group flex min-h-36 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-6 text-center transition ${
            isDragging
              ? 'border-green-700 bg-green-100 text-green-900'
              : 'border-slate-300/70 bg-white text-slate-600 hover:border-green-700 hover:text-slate-900'
          }`}
        >
          <UploadCloud className="h-7 w-7" />
          <div>
            <p className="text-sm font-medium">
              {file ? file.name : 'Drag and drop a .wav file, or click to browse'}
            </p>
            <p className="text-xs opacity-80">16kHz mono or stereo WAV recommended</p>
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

          <button
            type="button"
            onClick={analyzeFile}
            disabled={isLoading || !file}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-green-700 px-5 text-sm font-semibold text-white transition hover:bg-green-600 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing engine sound...
              </>
            ) : (
              <>
                <Activity className="h-4 w-4" />
                Analyze
              </>
            )}
          </button>
        </div>

        {error ? (
          <div className="rounded-lg border border-amber-400/35 bg-amber-50 p-3 text-sm text-amber-900">
            <div className="mb-1 inline-flex items-center gap-2 font-medium">
              <AlertTriangle className="h-4 w-4" />
              Analysis failed
            </div>
            <p>{error}</p>
          </div>
        ) : null}

        {result ? (
          <div className="space-y-4 rounded-xl border border-green-800/30 bg-white p-4">
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
                  className="h-2 rounded-full bg-linear-to-r from-green-800 via-green-600 to-green-700"
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
            </div>

            <p className="rounded-md border border-green-800/25 bg-green-50 p-3 text-sm text-slate-800">
              {result.notes}
            </p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
