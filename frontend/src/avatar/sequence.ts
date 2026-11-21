import { NUM_POINTS } from './layout';

export interface SentenceBoundary {
  index: number;
  startFrame: number;
  endFrame: number;
  text: string;
}

export interface PoseSequenceData {
  id: string;
  title: string;
  fps: number;
  frames: Float32Array;
  frameCount: number;
  sentences: SentenceBoundary[];
  source: 'real' | 'synthetic' | 'generated' | string;
  synthetic: boolean;
  warning: string;
  text: string;
  layout: string;
  normalization: string;
}

export interface DemoManifest {
  count: number;
  defaultSequenceId: string | null;
  sessionSequenceId: string | null;
  fps: number;
  layout: string;
  realDataAvailable: boolean;
  disclaimer: string;
  sequences: Array<Omit<PoseSequenceData, 'frames' | 'frameCount'>>;
}

export function frameCountOf(frames: Float32Array, points = NUM_POINTS): number {
  return Math.floor(frames.length / (points * 2));
}

export function getFrame(
  frames: Float32Array,
  index: number,
  out = new Float32Array(NUM_POINTS * 2),
  points = NUM_POINTS,
): Float32Array {
  const clamped = Math.max(0, Math.min(frameCountOf(frames, points) - 1, index));
  const offset = clamped * points * 2;
  out.set(frames.subarray(offset, offset + points * 2));
  return out;
}

/** Linear interpolation between two frames (positions only). */
export function lerpFrames(
  frames: Float32Array,
  a: number,
  b: number,
  alpha: number,
  out = new Float32Array(NUM_POINTS * 2),
  points = NUM_POINTS,
): Float32Array {
  const count = frameCountOf(frames, points);
  const ia = Math.max(0, Math.min(count - 1, a));
  const ib = Math.max(0, Math.min(count - 1, b));
  const oa = ia * points * 2;
  const ob = ib * points * 2;
  const t = Math.max(0, Math.min(1, alpha));
  for (let i = 0; i < points * 2; i++) {
    out[i] = frames[oa + i] + (frames[ob + i] - frames[oa + i]) * t;
  }
  return out;
}

/**
 * Concatenate sequences into one timeline. The last `crossfadeFrames` frames
 * of each clip are blended linearly into the first frames of the next clip,
 * which hides small discontinuities without inserting a rest pose.
 */
export function concatSequences(
  clips: PoseSequenceData[],
  crossfadeFrames = 7,
): PoseSequenceData {
  if (clips.length === 0) throw new Error('No clips to concatenate');
  const fade = Math.max(1, Math.min(crossfadeFrames, ...clips.map((c) => c.frameCount - 1)));
  const total = clips.reduce((n, c) => n + c.frameCount, 0) - fade * (clips.length - 1);
  const frames = new Float32Array(total * NUM_POINTS * 2);
  const sentences: SentenceBoundary[] = [];
  let write = 0;
  let sentenceOffset = 0;

  clips.forEach((clip, clipIndex) => {
    for (let f = 0; f < clip.frameCount; f++) {
      const outFrame = write - (clipIndex > 0 ? fade : 0) + f;
      if (outFrame < 0 || outFrame >= total) continue;
      const dst = outFrame * NUM_POINTS * 2;
      const src = f * NUM_POINTS * 2;
      if (clipIndex > 0 && f < fade) {
        // Blend with the tail of the previous clip already written here.
        const w = (f + 1) / (fade + 1);
        for (let i = 0; i < NUM_POINTS * 2; i++) {
          frames[dst + i] = frames[dst + i] * (1 - w) + clip.frames[src + i] * w;
        }
      } else {
        frames.set(clip.frames.subarray(src, src + NUM_POINTS * 2), dst);
      }
    }
    for (const s of clip.sentences) {
      sentences.push({
        ...s,
        index: sentenceOffset + clip.sentences.indexOf(s),
        startFrame: s.startFrame + write - (clipIndex > 0 ? fade : 0),
        endFrame: s.endFrame + write - (clipIndex > 0 ? fade : 0),
      });
    }
    sentenceOffset += clip.sentences.length;
    write += clip.frameCount;
  });

  const first = clips[0];
  return {
    id: `session_${clips.map((c) => c.id).join('_')}`,
    title: `Session — ${clips.length} clips`,
    fps: first.fps,
    frames,
    frameCount: total,
    sentences,
    source: clips.some((c) => c.source === 'real') ? 'real' : 'synthetic',
    synthetic: clips.every((c) => c.synthetic),
    warning: first.warning,
    text: sentences.map((s) => s.text).join(' '),
    layout: first.layout,
    normalization: first.normalization,
  };
}
