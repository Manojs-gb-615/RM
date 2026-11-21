import type { Quaternion } from 'three';
import type { FaceChannels } from './faceChannels';

export interface PoseMeta {
  depthFlips: number;
  handConfidence: number;
  elbowClamped: number;
  headFitResidual: number;
}

/** One solved frame: local bone quaternions plus face channels. */
export interface PoseFrame {
  locals: Map<string, Quaternion>;
  face: FaceChannels;
  meta: PoseMeta;
}

export interface KeypointFrame {
  /** [137 * 2] canonical x,y in shoulder-width units, y up. */
  data: Float32Array;
  frameIndex: number;
}
