import type { Layout } from './layout';

/**
 * All tunables for the browser-side pipeline. Every value here is exposed in
 * the demo debug panel; changing it must not require a backend round-trip.
 */
export interface AvatarConfig {
  /** Layout JSON imported by `layout.ts` (canonical point indices). */
  layout: Layout;
  /** Mirror left/right data (off by default: anatomical mapping). */
  mirror: boolean;

  filter: {
    body: OneEuroConfig;
    hands: OneEuroConfig;
    face: OneEuroConfig;
    /** Gaps up to this many frames are interpolated; longer gaps hold. */
    maxGapFrames: number;
    /** Joint speed above this (shoulder widths/frame) is an outlier spike. */
    maxJointSpeed: number;
  };

  depth: {
    /** Prior weight for upper arms pointing toward the camera. */
    upperArmForward: number;
    /** Prior weight for forearms/hands pointing toward the camera. */
    forearmForward: number;
    /** z advantage required before a depth sign flip is accepted. */
    depthFlipHysteresis: number;
    /** Minimum elbow flexion in degrees. */
    minElbowFlexion: number;
    /** Maximum elbow flexion in degrees. */
    maxElbowFlexion: number;
    /** Extra z (shoulder widths) applied to the whole arm as a signing plane. */
    signingPlaneZ: number;
  };

  hands: {
    /** Sign of the palm normal for the anatomical left hand. */
    palmSignLeft: number;
    /** Sign of the palm normal for the anatomical right hand. */
    palmSignRight: number;
    /** Extra roll (radians) applied to the hand bone after forward alignment. */
    rollLeft: number;
    rollRight: number;
    /** Cap on finger curl (radians) used to bend unresolved joints. */
    maxCurl: number;
    /** Fall back to the opposite palm sign when this frame is ambiguous. */
    ambiguityThreshold: number;
  };

  torso: {
    rollGain: number;
    yawGain: number;
    pitchGain: number;
    maxRoll: number;
    maxYaw: number;
    maxPitch: number;
  };

  head: {
    enabled: boolean;
    /** Blend factor between the fitted rotation and neutral. */
    gain: number;
    maxYaw: number;
    maxPitch: number;
    maxRoll: number;
    /** Number of Gauss-Newton iterations for the landmark fit. */
    iterations: number;
  };

  smoothing: {
    /** Rotation half-life in seconds for the body. */
    bodyHalfLife: number;
    /** Rotation half-life in seconds for hands (shorter = more responsive). */
    handHalfLife: number;
    /** At sentence boundaries, crossfade this many data frames (6-8). */
    boundaryCrossfadeFrames: number;
  };

  face: {
    /** Frames used for neutral-face calibration. */
    neutralFrames: number;
    gain: {
      browRaise: number;
      browFurrow: number;
      blink: number;
      jawOpen: number;
      mouthWide: number;
      mouthRound: number;
      smile: number;
      frown: number;
      gaze: number;
    };
    smoothingHalfLife: number;
    /** Show a warning when the avatar has no brow controls. */
    warnOnMissingBrows: boolean;
  };

  camera: {
    /** Vertical field of view in degrees. */
    fov: number;
    /** Camera distance in shoulder-width units. */
    distance: number;
    /** Vertical framing offset in shoulder-width units. */
    targetY: number;
    /** Automatically put the camera on the side the avatar faces. */
    autoFacing: boolean;
    /** Manual yaw in degrees (used when autoFacing is off). */
    yawDeg: number;
  };

  render: {
    background: string;
    skinTone: string;
    showKeypoints: boolean;
    showStickFigure: boolean;
  };
}

export interface OneEuroConfig {
  minCutoff: number;
  beta: number;
  dCutoff: number;
}

export const DEFAULT_CONFIG: AvatarConfig = {
  // Replaced by `createConfig()` with the imported layout.
  layout: null as unknown as Layout,
  mirror: false,
  filter: {
    body: { minCutoff: 1.2, beta: 0.02, dCutoff: 1.0 },
    // Hands change shape fast; a higher cutoff keeps fingerspelling readable.
    hands: { minCutoff: 3.5, beta: 0.05, dCutoff: 1.0 },
    face: { minCutoff: 2.0, beta: 0.01, dCutoff: 1.0 },
    maxGapFrames: 5,
    maxJointSpeed: 1.6,
  },
  depth: {
    upperArmForward: 0.25,
    forearmForward: 1.0,
    depthFlipHysteresis: 0.35,
    minElbowFlexion: 5,
    maxElbowFlexion: 175,
    signingPlaneZ: 0.12,
  },
  hands: {
    palmSignLeft: 1,
    palmSignRight: -1,
    rollLeft: 0,
    rollRight: 0,
    maxCurl: Math.PI * 0.95,
    ambiguityThreshold: 0.02,
  },
  torso: {
    rollGain: 1.0,
    yawGain: 0.6,
    pitchGain: 0.5,
    maxRoll: 0.25,
    maxYaw: 0.18,
    maxPitch: 0.12,
  },
  head: {
    enabled: true,
    gain: 0.8,
    maxYaw: 0.45,
    maxPitch: 0.35,
    maxRoll: 0.35,
    iterations: 14,
  },
  smoothing: {
    bodyHalfLife: 0.06,
    handHalfLife: 0.025,
    boundaryCrossfadeFrames: 7,
  },
  face: {
    neutralFrames: 30,
    gain: {
      browRaise: 1.0,
      browFurrow: 1.0,
      blink: 1.0,
      jawOpen: 1.0,
      mouthWide: 1.0,
      mouthRound: 1.0,
      smile: 1.0,
      frown: 1.0,
      gaze: 1.0,
    },
    smoothingHalfLife: 0.04,
    warnOnMissingBrows: true,
  },
  camera: {
    fov: 32,
    distance: 4.6,
    targetY: -0.35,
    autoFacing: true,
    yawDeg: 0,
  },
  render: {
    background: '#181c24',
    skinTone: '#c98d6b',
    showKeypoints: false,
    showStickFigure: true,
  },
};

export function cloneConfig(config: AvatarConfig): AvatarConfig {
  return JSON.parse(JSON.stringify(config)) as AvatarConfig;
}
