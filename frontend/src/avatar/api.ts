import { NUM_POINTS } from './layout';
import {
  frameCountOf,
  type DemoManifest,
  type PoseSequenceData,
  type SentenceBoundary,
} from './sequence';

export interface CsapHeader {
  format: string;
  version: number;
  sequenceId: string;
  fps: number;
  layout: string;
  T: number;
  points: number;
  dims: number;
  dtype: string;
  normalization: string;
  source: string;
  synthetic?: boolean;
  title?: string;
  warning?: string;
  text?: string;
  sentences?: SentenceBoundary[];
  disclaimer?: string;
}

export const API_BASE: string =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE ??
  'http://localhost:8000';

/** Decode the CSAP binary payload documented in backend/services/canonical_pose.py. */
export function decodeCsap(buffer: ArrayBuffer): { header: CsapHeader; frames: Float32Array } {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12 || String.fromCharCode(...bytes.slice(0, 4)) !== 'CSAP') {
    throw new Error('Not a CSAP pose payload');
  }
  const view = new DataView(buffer);
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported CSAP version ${version}`);
  const headerLength = view.getUint32(8, true);
  const headerText = new TextDecoder().decode(bytes.slice(12, 12 + headerLength));
  const header = JSON.parse(headerText) as CsapHeader;
  const payloadOffset = 12 + headerLength;
  const expected = header.T * NUM_POINTS * 2 * 4;
  if (bytes.length - payloadOffset !== expected) {
    throw new Error(
      `CSAP payload length ${bytes.length - payloadOffset} does not match header (${expected})`,
    );
  }
  const frames = new Float32Array(buffer.slice(payloadOffset));
  return { header, frames };
}

export async function fetchManifest(base = API_BASE): Promise<DemoManifest> {
  const response = await fetch(`${base}/api/poses/demo`);
  if (!response.ok) throw new Error(`Demo manifest request failed (${response.status})`);
  return (await response.json()) as DemoManifest;
}

export async function fetchSequence(
  sequenceId: string,
  base = API_BASE,
): Promise<PoseSequenceData> {
  const response = await fetch(
    `${base}/api/poses/${encodeURIComponent(sequenceId)}?format=bin`,
  );
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(`Pose request failed (${response.status}) ${detail}`);
  }
  const { header, frames } = decodeCsap(await response.arrayBuffer());
  return {
    id: sequenceId,
    title: header.title ?? sequenceId,
    fps: header.fps || 24,
    frames,
    frameCount: frameCountOf(frames),
    sentences: header.sentences ?? [],
    source: header.source ?? 'synthetic',
    synthetic: header.synthetic ?? header.source === 'synthetic',
    warning: header.warning ?? '',
    text: header.text ?? '',
    layout: header.layout,
    normalization: header.normalization,
  };
}

export async function fetchSequenceJson(
  sequenceId: string,
  base = API_BASE,
): Promise<PoseSequenceData> {
  const response = await fetch(`${base}/api/poses/${encodeURIComponent(sequenceId)}`);
  if (!response.ok) throw new Error(`Pose request failed (${response.status})`);
  const data = (await response.json()) as {
    header: CsapHeader;
    points: number[][][];
    sentences?: SentenceBoundary[];
  };
  const flat: number[] = [];
  for (const frame of data.points) {
    for (const point of frame) {
      flat.push(point[0], point[1]);
    }
  }
  const frames = new Float32Array(flat);
  return {
    id: sequenceId,
    title: data.header.title ?? sequenceId,
    fps: data.header.fps || 24,
    frames,
    frameCount: frameCountOf(frames),
    sentences: data.sentences ?? [],
    source: data.header.source ?? 'synthetic',
    synthetic: data.header.synthetic ?? data.header.source === 'synthetic',
    warning: data.header.warning ?? '',
    text: data.header.text ?? '',
    layout: data.header.layout,
    normalization: data.header.normalization,
  };
}
