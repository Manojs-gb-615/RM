import type { AvatarConfig, OneEuroConfig } from './config';
import { NUM_POINTS } from './layout';
import { clamp } from './math3';

/** Standard One Euro filter for a scalar signal. */
export class OneEuroFilter {
  private hatXPrev = 0;
  private dxPrev = 0;
  private initialized = false;

  constructor(private readonly cfg: OneEuroConfig) {}

  reset(): void {
    this.initialized = false;
  }

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value: number, dt: number): number {
    if (!this.initialized) {
      this.initialized = true;
      this.hatXPrev = value;
      this.dxPrev = 0;
      return value;
    }
    const dx = (value - this.hatXPrev) / dt;
    const aD = this.alpha(this.cfg.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const cutoff = this.cfg.minCutoff + this.cfg.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff, dt);
    const hatX = a * value + (1 - a) * this.hatXPrev;
    this.dxPrev = dxHat;
    this.hatXPrev = hatX;
    return hatX;
  }
}

export interface FilteredSequence {
  /** [T, 137, 2] filtered canonical keypoints. */
  frames: Float32Array;
  /** Per-frame count of points rejected as speed outliers. */
  outliers: Uint16Array;
  /** Per-frame count of points repaired from missing values. */
  repaired: Uint16Array;
  /** Raw frames after gap/outlier repair but before smoothing (for jitter tests). */
  repairedFrames: Float32Array;
}

function groupRanges(): Array<{ start: number; count: number; cfg: 'body' | 'hands' | 'face' }> {
  return [
    { start: 0, count: 25, cfg: 'body' },
    { start: 25, count: 70, cfg: 'face' },
    { start: 95, count: 21, cfg: 'hands' },
    { start: 116, count: 21, cfg: 'hands' },
  ];
}

/**
 * Repair missing values, reject implausible joint speeds, then apply a
 * One Euro filter with separate parameters for body, hands and face.
 *
 * The backend adapter already interpolates short gaps; this is the browser's
 * second line of defence (and it is also used by synthetic data, which has no
 * gaps at all).
 */
export function filterSequence(
  input: Float32Array,
  fps: number,
  config: AvatarConfig,
): FilteredSequence {
  const T = input.length / (NUM_POINTS * 2);
  const dt = 1 / Math.max(1e-6, fps);
  const repairedFrames = new Float32Array(input);
  const outliers = new Uint16Array(T);
  const repaired = new Uint16Array(T);

  // 1. Missing-value repair. Canonical space uses (0,0) for the neck, so a
  // point is only "missing" here if it is non-finite; the backend already
  // distinguishes OpenPose sentinels from valid zero coordinates.
  for (let p = 0; p < NUM_POINTS; p++) {
    let lastX = Number.NaN;
    let lastY = Number.NaN;
    let gap = 0;
    const firstValid = findFirstValid(input, T, p);
    for (let t = 0; t < T; t++) {
      const o = (t * NUM_POINTS + p) * 2;
      const x = input[o];
      const y = input[o + 1];
      if (Number.isFinite(x) && Number.isFinite(y)) {
        lastX = x;
        lastY = y;
        gap = 0;
      } else {
        const fallbackX = Number.isFinite(lastX) ? lastX : firstValid.x;
        const fallbackY = Number.isFinite(lastY) ? lastY : firstValid.y;
        repairedFrames[o] = fallbackX;
        repairedFrames[o + 1] = fallbackY;
        gap++;
        if (gap > config.filter.maxGapFrames) {
          // Hold the last value; nothing else to do because it is already held.
        }
        repaired[t]++;
      }
    }
  }

  // 2. Outlier rejection: a joint moving faster than maxJointSpeed in one
  // frame is a detection spike, not signing motion.
  const prev = new Float32Array(NUM_POINTS * 2);
  prev.set(repairedFrames.subarray(0, NUM_POINTS * 2));
  for (let t = 1; t < T; t++) {
    for (let p = 0; p < NUM_POINTS; p++) {
      const o = (t * NUM_POINTS + p) * 2;
      const dx = repairedFrames[o] - prev[p * 2];
      const dy = repairedFrames[o + 1] - prev[p * 2 + 1];
      const speed = Math.hypot(dx, dy);
      if (speed > config.filter.maxJointSpeed) {
        repairedFrames[o] = prev[p * 2];
        repairedFrames[o + 1] = prev[p * 2 + 1];
        outliers[t]++;
      }
      prev[p * 2] = repairedFrames[o];
      prev[p * 2 + 1] = repairedFrames[o + 1];
    }
  }

  // 3. One Euro filtering, one filter per coordinate per region.
  const frames = new Float32Array(repairedFrames);
  for (const group of groupRanges()) {
    const cfg = config.filter[group.cfg];
    const filters: OneEuroFilter[] = [];
    for (let i = 0; i < group.count * 2; i++) filters.push(new OneEuroFilter(cfg));
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < group.count; i++) {
        const point = group.start + i;
        const o = (t * NUM_POINTS + point) * 2;
        frames[o] = filters[i * 2].filter(repairedFrames[o], dt);
        frames[o + 1] = filters[i * 2 + 1].filter(repairedFrames[o + 1], dt);
      }
    }
  }

  return { frames, outliers, repaired, repairedFrames };
}

function findFirstValid(input: Float32Array, T: number, point: number): { x: number; y: number } {
  for (let t = 0; t < T; t++) {
    const o = (t * NUM_POINTS + point) * 2;
    if (Number.isFinite(input[o]) && Number.isFinite(input[o + 1])) {
      return { x: input[o], y: input[o + 1] };
    }
  }
  return { x: 0, y: 0 };
}

/**
 * Clamp keypoint values to a sane region around the body. Detections far
 * outside the frame otherwise produce violent avatar rotations.
 */
export function clampOutliers(input: Float32Array, limit = 4): Float32Array {
  const out = new Float32Array(input);
  for (let i = 0; i < out.length; i++) out[i] = clamp(out[i], -limit, limit);
  return out;
}
