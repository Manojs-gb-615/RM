import { Matrix4, Quaternion, Vector3 } from 'three';

export const UP = new Vector3(0, 1, 0);
export const FORWARD = new Vector3(0, 0, 1);

export function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Minimal rotation taking `from` to `to`; both must be unit length. */
export function alignQuaternion(
  from: Vector3,
  to: Vector3,
  out = new Quaternion(),
): Quaternion {
  return out.setFromUnitVectors(from, to);
}

/** Rotation matrix from a right-handed orthonormal basis. */
export function quatFromBasis(
  xAxis: Vector3,
  yAxis: Vector3,
  zAxis: Vector3,
  out = new Quaternion(),
): Quaternion {
  const m = new Matrix4().makeBasis(xAxis, yAxis, zAxis);
  return out.setFromRotationMatrix(m);
}

/** Rotate `dir` around `axis` by `angle` radians. */
export function rotateAround(
  dir: Vector3,
  axis: Vector3,
  angle: number,
  out = new Vector3(),
): Vector3 {
  return out.copy(dir).applyAxisAngle(axis, angle);
}

/** Quaternion exponential-ish smoothing: `alpha = 1 - 2^(-dt/halfLife)`. */
export function halfLifeAlpha(dt: number, halfLife: number): number {
  if (halfLife <= 0) return 1;
  return 1 - Math.pow(2, -dt / halfLife);
}

export function safeNormalize(v: Vector3, fallback = new Vector3(1, 0, 0)): Vector3 {
  const len = v.length();
  if (len < 1e-8 || !Number.isFinite(len)) return v.copy(fallback);
  return v.multiplyScalar(1 / len);
}

/** Angle between two unit vectors, in radians. */
export function angleBetween(a: Vector3, b: Vector3): number {
  return Math.acos(clamp(a.dot(b), -1, 1));
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
