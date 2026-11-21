import { Quaternion, Vector3 } from 'three';
import type { AvatarConfig } from './config';
import { HAND_LOCAL, handPoint } from './layout';
import { quatFromBasis, safeNormalize } from './math3';
import { FINGERS, fingerBoneNames, type Side, type Skeleton } from './skeleton';

export interface HandSolution {
  /** World/rest-space orientation for the hand bone. */
  handQuat: Quaternion;
  /** Target world directions for thumb/index/middle/ring/little bones. */
  fingerDirs: Map<string, Vector3>;
  /** Mean curl 0..1 (0 = straight, 1 = fully folded). */
  curl: number;
  /** 1 when the palm sign is confident this frame. */
  confidence: number;
  /** True when the palm normal flipped relative to the previous frame. */
  flipped: boolean;
  /** True when the hand keypoints were unusable and a neutral hand was used. */
  neutral: boolean;
}

interface HandState {
  palmFacing: number;
  wristDir: Vector3;
}

const NEUTRAL: HandState = { palmFacing: 0, wristDir: new Vector3() };

function point2(frame: Float32Array, index: number): { x: number; y: number } {
  return { x: frame[index * 2], y: frame[index * 2 + 1] };
}

/**
 * Hand orientation and per-finger target directions from the 21-point hand.
 *
 * Approximations (documented in DECISIONS.md):
 *  - The palm normal is inferred from the signed 2D area of
 *    (wrist, indexMCP, pinkyMCP) combined with handedness.
 *  - Finger depth comes from the difference between projected segment length
 *    and the avatar's rest segment length; the residual is assigned to the
 *    palm normal direction.
 */
export function solveHand(
  frame: Float32Array,
  handStart: number,
  palmSide: Side,
  skeletonSide: Side,
  foreDir: Vector3,
  skeleton: Skeleton,
  config: AvatarConfig,
  state: HandState = NEUTRAL,
  xSign = 1,
): HandSolution {
  const W = point2(frame, handPoint(handStart, HAND_LOCAL.wrist));
  const I = point2(frame, handPoint(handStart, HAND_LOCAL.indexMcp));
  const P = point2(frame, handPoint(handStart, HAND_LOCAL.pinkyMcp));
  const M = point2(frame, handPoint(handStart, HAND_LOCAL.middleMcp));

  const forward2 = { x: (M.x - W.x) * xSign, y: M.y - W.y };
  const lateral2 = { x: (I.x - P.x) * xSign, y: I.y - P.y };
  const area = forward2.x * lateral2.y - forward2.y * lateral2.x;
  const handScale = Math.hypot(forward2.x, forward2.y) + 0.15;
  const confident = Math.abs(area) > config.hands.ambiguityThreshold * handScale * handScale;

  const palmSign = palmSide === 'left' ? config.hands.palmSignLeft : config.hands.palmSignRight;
  let palmFacing = confident ? Math.sign(area) * palmSign : state.palmFacing || palmSign;
  if (!Number.isFinite(palmFacing) || palmFacing === 0) palmFacing = palmSign;

  const handQuat = new Quaternion();
  const fingerDirs = new Map<string, Vector3>();

  const fDir = new Vector3(forward2.x, forward2.y, 0);
  if (fDir.lengthSq() < 1e-8) fDir.copy(foreDir);
  fDir.normalize();
  // Keep the fingers in front of the palm plane by a small amount; this is a
  // visual prior, not a recovered quantity.
  fDir.z += 0.05;
  fDir.normalize();

  const lateral = new Vector3(lateral2.x, lateral2.y, 0);
  if (lateral.lengthSq() < 1e-8) lateral.set(1, 0, 0);
  lateral.normalize();
  const normal = new Vector3(0, 0, palmFacing);
  // Keep the basis right-handed: y = z × x. The projected lateral direction
  // (index MCP - pinky MCP) disambiguates which way the palm normal points;
  // flipping the normal is the only freedom that preserves handedness.
  const lateralOrtho = new Vector3().crossVectors(normal, fDir).normalize();
  if (lateralOrtho.dot(lateral) < 0) {
    palmFacing = -palmFacing;
    normal.set(0, 0, palmFacing);
    lateralOrtho.crossVectors(normal, fDir).normalize();
  }
  if (lateralOrtho.lengthSq() < 1e-8) lateralOrtho.copy(lateral);

  // For a hand with palm toward the camera, x=forward, y=lateral, z=normal is
  // right-handed. Rest basis is derived from the avatar's own rest fingers.
  const rest = restHandBasis(skeleton, skeletonSide);
  const target = quatFromBasis(
    new Vector3().copy(fDir),
    lateralOrtho.clone(),
    new Vector3(0, 0, palmFacing),
  );
  handQuat.copy(target).multiply(rest.clone().invert()).normalize();

  let curlSum = 0;
  let curlCount = 0;
  const palmDepthSign = palmFacing >= 0 ? 1 : -1;

  for (const finger of FINGERS) {
    const key = finger.toLowerCase() as keyof typeof HAND_LOCAL;
    const locals = HAND_LOCAL[key] as unknown as number[] | undefined;
    if (!Array.isArray(locals) || locals.length === 0) continue;
    // HAND_LOCAL stores the first and last OpenPose point of each finger. The
    // points are contiguous, and every VRM finger chain has exactly three
    // bones, so segment i runs from local point (first + i) to (first + i + 1).
    const base = locals[0];
    const bones = fingerBoneNames(skeletonSide, finger);
    for (let i = 0; i < bones.length; i++) {
      const a = point2(frame, handPoint(handStart, base + i));
      const b = point2(frame, handPoint(handStart, base + i + 1));
      const dx = (b.x - a.x) * xSign;
      const dy = b.y - a.y;
      const projected = Math.hypot(dx, dy);
      const joint = skeleton.joints.get(bones[i]);
      const len = joint?.length ?? 0;
      const dir = new Vector3(dx, dy, 0);
      if (len <= 1e-6) {
        dir.copy(fDir);
      } else if (projected >= len) {
        dir.set(dx, dy, 0).multiplyScalar(len / Math.max(projected, 1e-6));
      } else {
        const dz = Math.sqrt(Math.max(0, len * len - projected * projected));
        dir.set(dx, dy, dz * palmDepthSign);
      }
      if (dir.lengthSq() < 1e-10) dir.copy(fDir);
      dir.normalize();
      // Blend against the observed 2D direction to avoid a bone pointing
      // backwards when the residual is noisy.
      const planar = new Vector3(dx, dy, 0);
      if (planar.lengthSq() > 1e-10) {
        planar.normalize();
        dir.lerp(planar, 0.15);
        dir.normalize();
      }
      fingerDirs.set(bones[i], dir);
      if (len > 1e-6) {
        curlSum += 1 - Math.min(1, projected / len);
        curlCount++;
      }
    }
  }

  const flipped = state.palmFacing !== 0 && state.palmFacing !== palmFacing;
  state.palmFacing = palmFacing;
  state.wristDir = new Vector3().copy(fDir);

  return {
    handQuat,
    fingerDirs,
    curl: curlCount ? curlSum / curlCount : 0,
    confidence: confident ? 1 : 0,
    flipped,
    neutral: false,
  };
}

function restHandBasis(skeleton: Skeleton, side: Side): Quaternion {
  const forward = skeleton.joints.get(`${side}IndexProximal`)?.pos
    ?.clone()
    .sub(skeleton.joints.get(`${side}Hand`)?.pos ?? new Vector3());
  const indexBase = skeleton.joints.get(`${side}IndexProximal`)?.pos;
  const pinkyBase = skeleton.joints.get(`${side}LittleProximal`)?.pos;
  const f = safeNormalize(forward ?? new Vector3(side === 'left' ? 1 : -1, 0, 0));
  const lateral = indexBase && pinkyBase ? safeNormalize(indexBase.clone().sub(pinkyBase)) : new Vector3(0, 0, 1);
  const n = new Vector3().crossVectors(f, lateral).normalize();
  const l = new Vector3().crossVectors(n, f).normalize();
  return quatFromBasis(f, l, n);
}

export function createHandState(): HandState {
  return { palmFacing: 0, wristDir: new Vector3() };
}

export type { HandState };
