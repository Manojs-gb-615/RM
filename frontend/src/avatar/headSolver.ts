import { Matrix4, Quaternion, Vector3 } from 'three';
import type { AvatarConfig } from './config';
import { FACE_LOCAL, IDX, facePoint } from './layout';
import { angleBetween, clamp } from './math3';

export interface HeadPose {
  quat: Quaternion;
  yaw: number;
  pitch: number;
  roll: number;
  residual: number;
  success: boolean;
}

interface ModelPoint {
  x: number;
  y: number;
  z: number;
  index: number;
}

const HEAD_PIVOT = new Vector3(0, 0.32, 0);

/**
 * Generic face geometry in shoulder-width units, y up, z toward the camera.
 * Deliberately a small stable set: nose, chin, eyes, ears and mouth.
 */
const MODEL: ModelPoint[] = [
  { x: 0.0, y: 0.40, z: 0.16, index: facePoint(FACE_LOCAL.noseTip) },
  { x: 0.0, y: 0.19, z: 0.07, index: facePoint(FACE_LOCAL.chin) },
  { x: -0.10, y: 0.47, z: 0.12, index: facePoint(FACE_LOCAL.rightEyeCenter) },
  { x: 0.10, y: 0.47, z: 0.12, index: facePoint(FACE_LOCAL.leftEyeCenter) },
  { x: -0.20, y: 0.42, z: 0.0, index: IDX.rightEar },
  { x: 0.20, y: 0.42, z: 0.0, index: IDX.leftEar },
  { x: 0.0, y: 0.28, z: 0.15, index: facePoint(FACE_LOCAL.mouthCenter) },
];

/**
 * Weak-perspective landmark fit for head rotation.
 *
 * A small Gauss-Newton optimisation over yaw/pitch/roll against a generic 3D
 * face model. It is deliberately low-dimensional: 2D landmarks cannot recover
 * head translation or fine facial depth, and an unconstrained fit is unstable
 * on noisy OpenPose faces.
 */
export function solveHead(frame: Float32Array, config: AvatarConfig, xSign = 1): HeadPose {
  const residualScale = 1;
  let yaw = 0;
  let pitch = 0;
  let roll = 0;

  if (!config.head.enabled) {
    return { quat: new Quaternion(), yaw, pitch, roll, residual: 0, success: false };
  }

  const iterations = Math.max(1, config.head.iterations);
  const lambda = 1e-4;

  for (let iter = 0; iter < iterations; iter++) {
    const jacobian: number[][] = [];
    const residuals: number[] = [];
    for (const point of MODEL) {
      const dx = frame[point.index * 2] * xSign;
      const dy = frame[point.index * 2 + 1];
      if (!Number.isFinite(dx) || !Number.isFinite(dy)) continue;
      const model = new Vector3(point.x, point.y, point.z).sub(HEAD_PIVOT);
      const projected = applyEuler(model, yaw, pitch, roll).add(HEAD_PIVOT);
      residuals.push((projected.x - dx) * residualScale, (projected.y - dy) * residualScale);

      const eps = 1e-3;
      const px = applyEuler(model, yaw + eps, pitch, roll).add(HEAD_PIVOT);
      const py = applyEuler(model, yaw, pitch + eps, roll).add(HEAD_PIVOT);
      const pz = applyEuler(model, yaw, pitch, roll + eps).add(HEAD_PIVOT);
      jacobian.push(
        [(px.x - projected.x) / eps, (py.x - projected.x) / eps, (pz.x - projected.x) / eps],
        [(px.y - projected.y) / eps, (py.y - projected.y) / eps, (pz.y - projected.y) / eps],
      );
    }
    if (residuals.length < 6) break;

    // Normal equations: (J^T J + lambda I) delta = -J^T r
    const A = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const b = [0, 0, 0];
    for (let r = 0; r < residuals.length; r++) {
      const row = jacobian[r];
      for (let i = 0; i < 3; i++) {
        b[i] += row[i] * residuals[r];
        for (let j = 0; j < 3; j++) A[i][j] += row[i] * row[j];
      }
    }
    for (let i = 0; i < 3; i++) {
      A[i][i] += lambda;
      b[i] = -b[i];
    }
    const delta = solve3x3(A, b);
    if (!delta) break;
    yaw = clamp(yaw + delta[0], -config.head.maxYaw, config.head.maxYaw);
    pitch = clamp(pitch + delta[1], -config.head.maxPitch, config.head.maxPitch);
    roll = clamp(roll + delta[2], -config.head.maxRoll, config.head.maxRoll);
    if (Math.hypot(delta[0], delta[1], delta[2]) < 1e-5) break;
  }

  let residual = 0;
  for (const point of MODEL) {
    const dx = frame[point.index * 2] * xSign;
    const dy = frame[point.index * 2 + 1];
    const projected = applyEuler(
      new Vector3(point.x, point.y, point.z).sub(HEAD_PIVOT),
      yaw,
      pitch,
      roll,
    ).add(HEAD_PIVOT);
    residual += Math.hypot(projected.x - dx, projected.y - dy);
  }
  residual /= MODEL.length;

  const quat = quaternionFromEuler(yaw, pitch, roll);
  return { quat, yaw, pitch, roll, residual, success: true };
}

function applyEuler(v: Vector3, yaw: number, pitch: number, roll: number): Vector3 {
  const m = matrixFromEuler(yaw, pitch, roll);
  return v.clone().applyMatrix4(m);
}

function matrixFromEuler(yaw: number, pitch: number, roll: number): Matrix4 {
  // R = Rz * Ry * Rx: yaw, then pitch, then roll (Three's 'YXZ' order).
  return new Matrix4()
    .makeRotationZ(roll)
    .multiply(new Matrix4().makeRotationY(yaw))
    .multiply(new Matrix4().makeRotationX(pitch));
}

function quaternionFromEuler(yaw: number, pitch: number, roll: number): Quaternion {
  return new Quaternion().setFromRotationMatrix(matrixFromEuler(yaw, pitch, roll));
}

function solve3x3(A: number[][], b: number[]): [number, number, number] | null {
  const m = [
    [A[0][0], A[0][1], A[0][2], b[0]],
    [A[1][0], A[1][1], A[1][2], b[1]],
    [A[2][0], A[2][1], A[2][2], b[2]],
  ];
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const div = m[col][col];
    for (let j = col; j < 4; j++) m[col][j] /= div;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = m[row][col];
      for (let j = col; j < 4; j++) m[row][j] -= factor * m[col][j];
    }
  }
  return [m[0][3], m[1][3], m[2][3]];
}

/** Eye-line roll and nose-offset yaw fallback when the fit is degenerate. */
export function fallbackHeadPose(frame: Float32Array, xSign = 1): HeadPose {
  const rightEye = new Vector3(frame[IDX.rightEye * 2] * xSign, frame[IDX.rightEye * 2 + 1], 0);
  const leftEye = new Vector3(frame[IDX.leftEye * 2] * xSign, frame[IDX.leftEye * 2 + 1], 0);
  const eyeLine = leftEye.clone().sub(rightEye);
  const roll = Math.atan2(eyeLine.y, eyeLine.x || 1e-6);
  const eyeMid = rightEye.clone().add(leftEye).multiplyScalar(0.5);
  const nose = new Vector3(frame[0] * xSign, frame[1], 0);
  const yaw = clamp((nose.x - eyeMid.x) * 2.5, -0.4, 0.4);
  const quat = quaternionFromEuler(yaw, 0, roll);
  return { quat, yaw, pitch: 0, roll, residual: 0, success: false };
}

export function headAngleDifference(a: HeadPose, b: HeadPose): number {
  return angleBetween(
    new Vector3(0, 0, 1).applyQuaternion(a.quat),
    new Vector3(0, 0, 1).applyQuaternion(b.quat),
  );
}
