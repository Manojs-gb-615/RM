// API client for SignLingo AI Mini backend

export interface StatusResponse {
  ready: boolean;
  num_indexed: number;
  dataset_samples: number;
  dataset_with_pose: number;
  is_synthetic: boolean;
  message: string;
}

export interface RetrievalResult {
  matched_sample_id: string | null;
  matched_sentence: string | null;
  score: number;
  pose: number[][] | null;   // [T, 274]
  no_match_reason: string | null;
  mode: string;
  ready: boolean;
  is_synthetic: boolean;
}

export interface CompareResult {
  current_sentence: string;
  text_only: RetrievalResult;
  context_aware: RetrievalResult;
  same_result: boolean;
}

export interface TranscriptSegment {
  index: number;
  start_seconds: number;
  end_seconds: number;
  text: string;
}

export interface ParsedTranscript {
  count: number;
  segments: TranscriptSegment[];
}

export interface RetrievedSegment {
  segment_index: number;
  start_seconds: number;
  end_seconds: number;
  sentence: string;
  retrieval: RetrievalResult;
}

export interface DatasetSample {
  sample_id: string;
  sentence: string;
  video_id: string;
  narrative_id: string;
  start_time: number;
  end_time: number;
  has_pose: boolean;
  prev_sample_id: string | null;
  prev_sentence: string | null;
  frames: number;
}

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  status: () => request<StatusResponse>('/status'),

  datasetSamples: () => request<{ count: number; samples: DatasetSample[] }>('/dataset/samples'),

  retrieve: (params: {
    current_sentence: string;
    mode: 'text_only' | 'context_aware';
    prev_sentence?: string;
    prev_pose?: number[][];
  }) =>
    request<RetrievalResult>('/retrieve', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  compare: (params: {
    current_sentence: string;
    prev_sentence?: string;
    prev_pose?: number[][];
  }) =>
    request<CompareResult>('/compare', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  parseTranscript: async (file: File): Promise<ParsedTranscript> => {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${BASE}/transcript/parse`, { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? `HTTP ${res.status}`);
    }
    return res.json();
  },

  retrieveAll: async (file: File, mode = 'context_aware'): Promise<{ mode: string; count: number; results: RetrievedSegment[] }> => {
    const fd = new FormData();
    fd.append('transcript_file', file);
    fd.append('mode', mode);
    const res = await fetch(`${BASE}/transcript/retrieve_all`, { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? `HTTP ${res.status}`);
    }
    return res.json();
  },
};
