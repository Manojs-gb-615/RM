import layoutJson from '../../../layouts/openpose137.json';
import { DEFAULT_CONFIG, type AvatarConfig } from './config';

export interface Region {
  start: number;
  count: number;
  names: string[];
}

export interface Layout {
  name: string;
  version: number;
  totalPoints: number;
  coordinateSpace: string;
  regions: {
    body: Region;
    face: Region;
    leftHand: Region;
    rightHand: Region;
  };
  indices: Record<string, number>;
  semantics: {
    faceLocal: Record<string, number | number[]>;
    handLocal: Record<string, number | number[]>;
  };
}

export const LAYOUT = layoutJson as unknown as Layout;
export const NUM_POINTS = LAYOUT.totalPoints;

/** Canonical point indices used across the renderer. */
export const IDX = {
  nose: 0,
  neck: 1,
  rightShoulder: 2,
  rightElbow: 3,
  rightWrist: 4,
  leftShoulder: 5,
  leftElbow: 6,
  leftWrist: 7,
  midHip: 8,
  rightHip: 9,
  leftHip: 12,
  rightEye: 15,
  leftEye: 16,
  rightEar: 17,
  leftEar: 18,
  faceStart: 25,
  leftHandStart: 95,
  rightHandStart: 116,
  rightPupil: 93,
  leftPupil: 94,
} as const;

export const FACE_LOCAL = {
  jaw: 0,
  rightBrow: 17,
  leftBrow: 22,
  noseBridge: 27,
  noseBase: 31,
  rightEye: 36,
  leftEye: 42,
  outerLips: 48,
  innerLips: 60,
  rightPupil: 68,
  leftPupil: 69,
  noseTip: 30,
  chin: 8,
  rightEyeCenter: 39,
  leftEyeCenter: 42,
  rightMouthCorner: 48,
  leftMouthCorner: 54,
  mouthCenter: 62,
  mouthUpperInner: 62,
  mouthLowerInner: 66,
} as const;

export const HAND_LOCAL = {
  wrist: 0,
  thumb: [1, 4],
  index: [5, 8],
  middle: [9, 12],
  ring: [13, 16],
  pinky: [17, 20],
  indexMcp: 5,
  middleMcp: 9,
  ringMcp: 13,
  pinkyMcp: 17,
  indexTip: 8,
  middleTip: 12,
  ringTip: 16,
  pinkyTip: 20,
  thumbTip: 4,
} as const;

export function facePoint(index: number): number {
  return IDX.faceStart + index;
}

export function handPoint(handStart: number, local: number): number {
  return handStart + local;
}

/** Body connections, in canonical indices (OpenPose BODY_25 topology). */
export const BODY_CONNECTIONS: Array<[number, number]> = [
  [1, 0],
  [1, 2],
  [2, 3],
  [3, 4],
  [1, 5],
  [5, 6],
  [6, 7],
  [1, 8],
  [8, 9],
  [9, 10],
  [10, 11],
  [8, 12],
  [12, 13],
  [13, 14],
  [0, 15],
  [15, 17],
  [0, 16],
  [16, 18],
];

export const HAND_CONNECTIONS: number[][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

export function createConfig(overrides: Partial<AvatarConfig> = {}): AvatarConfig {
  return {
    ...DEFAULT_CONFIG,
    ...overrides,
    layout: LAYOUT,
    filter: { ...DEFAULT_CONFIG.filter, ...(overrides.filter ?? {}) },
    depth: { ...DEFAULT_CONFIG.depth, ...(overrides.depth ?? {}) },
    hands: { ...DEFAULT_CONFIG.hands, ...(overrides.hands ?? {}) },
    torso: { ...DEFAULT_CONFIG.torso, ...(overrides.torso ?? {}) },
    head: { ...DEFAULT_CONFIG.head, ...(overrides.head ?? {}) },
    smoothing: { ...DEFAULT_CONFIG.smoothing, ...(overrides.smoothing ?? {}) },
    face: {
      ...DEFAULT_CONFIG.face,
      ...(overrides.face ?? {}),
      gain: { ...DEFAULT_CONFIG.face.gain, ...(overrides.face?.gain ?? {}) },
    },
    camera: { ...DEFAULT_CONFIG.camera, ...(overrides.camera ?? {}) },
    render: { ...DEFAULT_CONFIG.render, ...(overrides.render ?? {}) },
  };
}
