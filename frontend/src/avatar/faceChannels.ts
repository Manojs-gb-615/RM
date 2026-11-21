import { FACE_LOCAL, facePoint } from './layout';
import { clamp } from './math3';

export interface FaceChannels {
  /** Eyebrow raise in face-height units, calibrated to neutral. */
  browRaise: number;
  /** Eyebrow furrow in face-height units, calibrated to neutral. */
  browFurrow: number;
  /** 1 = eyes open, 0 = closed. */
  eyeOpen: number;
  /** Jaw opening in face-height units. */
  jawOpen: number;
  /** Mouth widening relative to neutral (smile-ish). */
  mouthWide: number;
  /** Mouth rounding/puckering relative to neutral. */
  mouthRound: number;
  /** Corner lift in face-height units. */
  smile: number;
  /** Corner drop in face-height units. */
  frown: number;
  /** Horizontal gaze, -1..1. */
  gazeX: number;
  /** Vertical gaze, -1..1. */
  gazeY: number;
}

export const NEUTRAL_FACE: FaceChannels = {
  browRaise: 0,
  browFurrow: 0,
  eyeOpen: 1,
  jawOpen: 0,
  mouthWide: 0,
  mouthRound: 0,
  smile: 0,
  frown: 0,
  gazeX: 0,
  gazeY: 0,
};

function range(frame: Float32Array, indices: number[], axis: 0 | 1): { min: number; max: number; mean: number } {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const i of indices) {
    const v = frame[i * 2 + axis];
    min = Math.min(min, v);
    max = Math.max(max, v);
    sum += v;
  }
  return { min, max, mean: sum / Math.max(1, indices.length) };
}

function center(frame: Float32Array, indices: number[]): { x: number; y: number } {
  const x = range(frame, indices, 0).mean;
  const y = range(frame, indices, 1).mean;
  return { x, y };
}

function span(a: number, b: number): number {
  return Math.abs(a - b);
}

/** Uncalibrated face features from the 70 canonical face points. */
export function extractFaceChannels(frame: Float32Array): FaceChannels {
  const f = (local: number) => facePoint(local);
  const rightBrow = Array.from({ length: 5 }, (_, i) => f(FACE_LOCAL.rightBrow + i));
  const leftBrow = Array.from({ length: 5 }, (_, i) => f(FACE_LOCAL.leftBrow + i));
  const rightEye = Array.from({ length: 6 }, (_, i) => f(FACE_LOCAL.rightEye + i));
  const leftEye = Array.from({ length: 6 }, (_, i) => f(FACE_LOCAL.leftEye + i));
  const outerLips = Array.from({ length: 12 }, (_, i) => f(FACE_LOCAL.outerLips + i));
  const innerLips = Array.from({ length: 8 }, (_, i) => f(FACE_LOCAL.innerLips + i));

  const rightEyeBox = { x: range(frame, rightEye, 0), y: range(frame, rightEye, 1) };
  const leftEyeBox = { x: range(frame, leftEye, 0), y: range(frame, leftEye, 1) };
  const rightEyeCenter = center(frame, rightEye);
  const leftEyeCenter = center(frame, leftEye);

  const eyeWidth = (rightEyeBox.x.max - rightEyeBox.x.min + leftEyeBox.x.max - leftEyeBox.x.min) / 2;
  const eyeHeight = (rightEyeBox.y.max - rightEyeBox.y.min + leftEyeBox.y.max - leftEyeBox.y.min) / 2;
  const interocular = span(leftEyeCenter.x, rightEyeCenter.x) || 1;
  const browY = (range(frame, rightBrow, 1).mean + range(frame, leftBrow, 1).mean) / 2;
  const eyeY = (rightEyeCenter.y + leftEyeCenter.y) / 2;
  const chinY = frame[f(FACE_LOCAL.chin) * 2 + 1];
  const faceHeight = Math.max(1e-4, span(browY, chinY));

  const browRaise = (browY - eyeY) / faceHeight;
  const browInnerGap = span(
    frame[rightBrow[rightBrow.length - 1] * 2],
    frame[leftBrow[0] * 2],
  );
  const browFurrow = 1 - clamp(browInnerGap / Math.max(interocular * 0.45, 1e-4), 0, 2);

  const eyeOpen = clamp(eyeHeight / Math.max(eyeWidth * 0.42, 1e-6), 0, 1.6);

  const innerBoxY = range(frame, innerLips, 1);
  const jawOpen = Math.max(0, innerBoxY.max - innerBoxY.min) / faceHeight;

  const outerX = range(frame, outerLips, 0);
  const mouthWidth = outerX.max - outerX.min;
  const mouthWide = mouthWidth / Math.max(interocular, 1e-4);
  const mouthRound = mouthWidth / Math.max(innerBoxY.max - innerBoxY.min + 1e-4, 1e-4);

  const cornerY =
    (frame[f(FACE_LOCAL.rightMouthCorner) * 2 + 1] +
      frame[f(FACE_LOCAL.leftMouthCorner) * 2 + 1]) /
    2;
  const mouthCenterY = (innerBoxY.max + innerBoxY.min) / 2;
  const cornerLift = (cornerY - mouthCenterY) / faceHeight;

  const pupilR = f(FACE_LOCAL.rightPupil);
  const pupilL = f(FACE_LOCAL.leftPupil);
  const gazeX =
    (frame[pupilL * 2] - leftEyeCenter.x + (frame[pupilR * 2] - rightEyeCenter.x)) /
    2 /
    Math.max(eyeWidth, 1e-4);
  const gazeY =
    (frame[pupilL * 2 + 1] - leftEyeCenter.y + (frame[pupilR * 2 + 1] - rightEyeCenter.y)) /
    2 /
    Math.max(eyeHeight, 1e-4);

  return {
    browRaise,
    browFurrow,
    eyeOpen,
    jawOpen,
    mouthWide,
    mouthRound,
    smile: Math.max(0, cornerLift),
    frown: Math.max(0, -cornerLift),
    gazeX: clamp(gazeX, -1, 1),
    gazeY: clamp(gazeY, -1, 1),
  };
}

const SCALAR_KEYS: Array<keyof FaceChannels> = [
  'browRaise',
  'browFurrow',
  'jawOpen',
  'mouthWide',
  'mouthRound',
  'smile',
  'frown',
  'gazeX',
  'gazeY',
];

/**
 * Neutral-face calibration: the median of the first N frames becomes the
 * baseline for every channel. ASL brow markers are relative to the signer's
 * neutral face, so this matters more than the absolute geometry.
 */
export class NeutralCalibrator {
  private samples: FaceChannels[] = [];
  private baseline: FaceChannels | null = null;

  constructor(private readonly neutralFrames: number) {}

  add(channels: FaceChannels): void {
    if (this.baseline || this.samples.length >= Math.max(1, this.neutralFrames)) return;
    this.samples.push(channels);
    if (this.samples.length >= Math.max(1, this.neutralFrames)) {
      const median: FaceChannels = { ...channels };
      for (const key of SCALAR_KEYS) {
        const values = this.samples.map((s) => s[key]).sort((a, b) => a - b);
        const mid = values.length >> 1;
        median[key] = values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
      }
      median.eyeOpen = Math.max(0.1, median.eyeOpen);
      this.baseline = median;
    }
  }

  get ready(): boolean {
    return this.baseline !== null;
  }

  get configuredFrames(): number {
    return this.neutralFrames;
  }

  reset(): void {
    this.samples = [];
    this.baseline = null;
  }

  apply(channels: FaceChannels): FaceChannels {
    const base = this.baseline;
    if (!base) return { ...channels };
    return {
      browRaise: channels.browRaise - base.browRaise,
      browFurrow: channels.browFurrow - base.browFurrow,
      eyeOpen: clamp(channels.eyeOpen / base.eyeOpen, 0, 1),
      jawOpen: Math.max(0, channels.jawOpen - base.jawOpen),
      mouthWide: channels.mouthWide / Math.max(base.mouthWide, 1e-4) - 1,
      mouthRound: channels.mouthRound / Math.max(base.mouthRound, 1e-4) - 1,
      smile: Math.max(0, channels.smile - base.smile),
      frown: Math.max(0, channels.frown - base.frown),
      gazeX: channels.gazeX - base.gazeX,
      gazeY: channels.gazeY - base.gazeY,
    };
  }
}
