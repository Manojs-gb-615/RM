import { Vector3 } from 'three';
import type { AvatarConfig } from './config';
import { clamp } from './math3';

export interface DepthSolution {
  /** Depth offset in shoulder-width units. */
  dz: number;
  /** True when the chosen sign differs from the previous frame. */
  flipped: boolean;
  /** True when the observed 2D length exceeded the bone length. */
  clamped: boolean;
}

/**
 * Weak-perspective depth from a known bone length and a 2D displacement.
 *
 * |dz| = sqrt(L^2 - d2^2), clamped when 2D noise makes d2 > L. The sign is
 * resolved with a "signing happens in front of the torso" prior plus temporal
 * continuity; a flip must beat the previous sign by `hysteresis` to be
 * accepted.
 */
export function solveDepth(
  dx: number,
  dy: number,
  length: number,
  previousZ: number,
  forwardPrior: number,
  hysteresis: number,
): DepthSolution {
  const d2 = Math.hypot(dx, dy);
  const safeLength = Math.max(length, 1e-6);
  const clampedD2 = Math.min(d2, safeLength * 0.9999);
  const magnitude = Math.sqrt(Math.max(0, safeLength * safeLength - clampedD2 * clampedD2));
  const clamped = d2 > safeLength;

  const prevSign = Math.sign(previousZ) || 1;
  // Lower score wins. Continuity dominates once a previous depth exists; the
  // forward prior (positive z) breaks ties and initials.
  const scoreFor = (sign: number) =>
    Math.abs(sign * magnitude - previousZ) - (sign > 0 ? forwardPrior * magnitude : 0);

  let sign = scoreFor(1) <= scoreFor(-1) ? 1 : -1;
  if (previousZ !== 0 && sign !== prevSign) {
    const advantage = Math.abs(scoreFor(sign) - scoreFor(prevSign));
    if (advantage < hysteresis) sign = prevSign;
  }
  return { dz: sign * magnitude, flipped: sign !== prevSign, clamped };
}

export interface ArmSolution {
  shoulder: Vector3;
  elbow: Vector3;
  wrist: Vector3;
  upperDir: Vector3;
  foreDir: Vector3;
  flips: number;
  clamped: number;
}

/**
 * Two-bone analytic solve for one arm in skeleton space.
 *
 * Positions are absolute (avatar rest shoulder + canonical offsets scaled by
 * the avatar shoulder width, which is 1 in skeleton space).
 */
export function solveArm(
  shoulder: Vector3,
  elbow2D: { x: number; y: number },
  wrist2D: { x: number; y: number },
  shoulder2D: { x: number; y: number },
  upperLength: number,
  foreLength: number,
  previous: { elbowZ: number; wristZ: number },
  config: AvatarConfig,
  out?: { elbow: Vector3; wrist: Vector3; upperDir: Vector3; foreDir: Vector3 },
): ArmSolution {
  const target = out ?? {
    elbow: new Vector3(),
    wrist: new Vector3(),
    upperDir: new Vector3(),
    foreDir: new Vector3(),
  };

  const e1x = (elbow2D.x - shoulder2D.x) * (config.mirror ? -1 : 1);
  const e1y = elbow2D.y - shoulder2D.y;
  const depth1 = solveDepth(
    e1x,
    e1y,
    upperLength,
    previous.elbowZ,
    config.depth.upperArmForward,
    config.depth.depthFlipHysteresis,
  );

  let upperDir = new Vector3(e1x, e1y, depth1.dz);
  if (upperDir.lengthSq() < 1e-10) upperDir.set(0, -1, 0);
  upperDir.normalize();

  // Apply the signing-plane offset along +z, then renormalize, so the hand
  // sits in front of the torso even in flat poses.
  upperDir.z = clamp(upperDir.z + config.depth.signingPlaneZ * 0.25, -1, 1);
  upperDir.normalize();

  const elbow = shoulder.clone().addScaledVector(upperDir, upperLength);

  const w2x = (wrist2D.x - elbow2D.x) * (config.mirror ? -1 : 1);
  const w2y = wrist2D.y - elbow2D.y;
  const depth2 = solveDepth(
    w2x,
    w2y,
    foreLength,
    previous.wristZ,
    config.depth.forearmForward,
    config.depth.depthFlipHysteresis,
  );
  let foreDir = new Vector3(w2x, w2y, depth2.dz);
  if (foreDir.lengthSq() < 1e-10) foreDir.copy(upperDir);
  foreDir.normalize();

  let clamped = 0;
  const clampedFore = clampElbowFlexion(
    upperDir,
    foreDir,
    (config.depth.minElbowFlexion * Math.PI) / 180,
    (config.depth.maxElbowFlexion * Math.PI) / 180,
  );
  if (clampedFore.clamped) clamped++;
  foreDir = clampedFore.dir;

  const wrist = elbow.clone().addScaledVector(foreDir, foreLength);

  target.elbow.copy(elbow);
  target.wrist.copy(wrist);
  target.upperDir.copy(upperDir);
  target.foreDir.copy(foreDir);

  return {
    shoulder,
    elbow: target.elbow,
    wrist: target.wrist,
    upperDir: target.upperDir,
    foreDir: target.foreDir,
    flips: (depth1.flipped ? 1 : 0) + (depth2.flipped ? 1 : 0),
    clamped: clamped + (depth1.clamped ? 1 : 0) + (depth2.clamped ? 1 : 0),
  };
}

/**
 * Clamp the angle between upper arm and forearm to a plausible elbow range.
 * The forearm direction is rebuilt in the plane it already spans, so the
 * wrist stays on the observed side unless the joint is at a limit.
 */
export function clampElbowFlexion(
  upperDir: Vector3,
  foreDir: Vector3,
  minAngle: number,
  maxAngle: number,
): { dir: Vector3; clamped: boolean } {
  const dot = clamp(upperDir.dot(foreDir), -1, 1);
  const angle = Math.acos(dot);
  if (angle >= minAngle && angle <= maxAngle) {
    return { dir: foreDir.clone(), clamped: false };
  }
  const target = clamp(angle, minAngle, maxAngle);
  const perpendicular = foreDir.clone().addScaledVector(upperDir, -dot);
  if (perpendicular.lengthSq() < 1e-10) {
    // Collinear: pick any perpendicular axis deterministically.
    perpendicular.set(0, 0, 1).cross(upperDir);
    if (perpendicular.lengthSq() < 1e-10) perpendicular.set(1, 0, 0);
  }
  perpendicular.normalize();
  const dir = upperDir
    .clone()
    .multiplyScalar(Math.cos(target))
    .addScaledVector(perpendicular, Math.sin(target))
    .normalize();
  return { dir, clamped: true };
}
