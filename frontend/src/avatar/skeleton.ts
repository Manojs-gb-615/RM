import { Quaternion, Vector3, type Object3D } from 'three';
import { safeNormalize } from './math3';

export interface RestJoint {
  name: string;
  parent: string | null;
  /** Rest position in skeleton space: neck origin, median shoulder width = 1. */
  pos: Vector3;
  /** Unit direction this bone points in at rest (pos - parent.pos). */
  dir: Vector3;
  /** Rest bone length in shoulder-width units. */
  length: number;
}

export interface Skeleton {
  joints: Map<string, RestJoint>;
  /** Parents before children. */
  order: string[];
  root: string;
  neckPos: Vector3;
  shoulderWidth: number;
  /** Human-readable source, e.g. "vrm" or "default". */
  source: string;
}

export const SIDES = ['left', 'right'] as const;
export type Side = (typeof SIDES)[number];

export const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'] as const;
export type Finger = (typeof FINGERS)[number];

const FINGER_BONES: Record<Finger, string[]> = {
  Thumb: ['ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal'],
  Index: ['IndexProximal', 'IndexIntermediate', 'IndexDistal'],
  Middle: ['MiddleProximal', 'MiddleIntermediate', 'MiddleDistal'],
  Ring: ['RingProximal', 'RingIntermediate', 'RingDistal'],
  Little: ['LittleProximal', 'LittleIntermediate', 'LittleDistal'],
};

export function fingerBoneNames(side: Side, finger: Finger): string[] {
  return FINGER_BONES[finger].map((bone) => `${side}${bone}`);
}

export function armBoneNames(side: Side): [string, string, string] {
  return [`${side}UpperArm`, `${side}LowerArm`, `${side}Hand`];
}

export function allFingerBones(): string[] {
  const out: string[] = [];
  for (const side of SIDES) {
    for (const finger of FINGERS) out.push(...fingerBoneNames(side, finger));
  }
  return out;
}

/**
 * Build a skeleton from absolute rest joint positions.
 *
 * Positions may be in any unit; they are re-expressed in the canonical
 * skeleton space (neck at origin, shoulder width 1). This is what makes the
 * same solver work for VRM avatars and the procedural mannequin.
 */
export function buildSkeleton(
  positions: Map<string, Vector3>,
  parentOf: Map<string, string | null>,
  shoulderLeft: string,
  shoulderRight: string,
  neck: string,
  source: string,
): Skeleton {
  const left = positions.get(shoulderLeft);
  const right = positions.get(shoulderRight);
  const neckPos = positions.get(neck);
  if (!left || !right || !neckPos) {
    throw new Error('Skeleton is missing shoulder or neck joints');
  }
  const shoulderWidth = left.distanceTo(right);
  if (!(shoulderWidth > 1e-6)) throw new Error('Degenerate shoulder width');

  const origin = neckPos.clone();
  const normalized = new Map<string, Vector3>();
  for (const [name, pos] of positions) {
    normalized.set(name, pos.clone().sub(origin).multiplyScalar(1 / shoulderWidth));
  }

  const joints = new Map<string, RestJoint>();
  for (const [name, pos] of normalized) {
    const parent = parentOf.get(name) ?? null;
    const parentPos = parent ? normalized.get(parent) : undefined;
    const delta = parentPos ? pos.clone().sub(parentPos) : new Vector3(0, 1, 0);
    const length = delta.length();
    joints.set(name, {
      name,
      parent,
      pos,
      dir: safeNormalize(delta.clone()),
      length,
    });
  }

  const order: string[] = [];
  const visiting = new Set<string>();
  const visit = (name: string) => {
    if (order.includes(name)) return;
    if (visiting.has(name)) throw new Error(`Cycle in skeleton at ${name}`);
    visiting.add(name);
    const joint = joints.get(name);
    if (joint?.parent) visit(joint.parent);
    visiting.delete(name);
    order.push(name);
  };
  for (const name of joints.keys()) visit(name);

  const root = order.find((name) => !joints.get(name)?.parent) ?? order[0];
  return {
    joints,
    order,
    root,
    neckPos: new Vector3(0, 0, 0),
    shoulderWidth: 1,
    source,
  };
}

/** Skeleton derived from live Three.js bone nodes (VRM normalized humanoid). */
export function skeletonFromNodes(
  nodes: Map<string, Object3D>,
  required: string[],
): Skeleton {
  nodes.forEach((node) => node.updateWorldMatrix(true, false));
  const positions = new Map<string, Vector3>();
  const parentOf = new Map<string, string | null>();
  const nodeToName = new Map<Object3D, string>();
  nodes.forEach((node, name) => nodeToName.set(node, name));

  for (const [name, node] of nodes) {
    positions.set(name, node.getWorldPosition(new Vector3()));
    let parentName: string | null = null;
    let cursor: Object3D | null = node.parent;
    while (cursor) {
      const found = nodeToName.get(cursor);
      if (found) {
        parentName = found;
        break;
      }
      cursor = cursor.parent;
    }
    parentOf.set(name, parentName);
  }

  for (const name of required) {
    if (!positions.has(name)) throw new Error(`Avatar is missing humanoid bone ${name}`);
  }
  return buildSkeleton(positions, parentOf, 'leftUpperArm', 'rightUpperArm', 'neck', 'vrm');
}

/**
 * Procedural T-pose skeleton, in canonical units. Used by the fallback
 * mannequin and by the evaluation harness so both exercise identical math.
 */
export function createDefaultSkeleton(): Skeleton {
  const positions = new Map<string, Vector3>();
  const parentOf = new Map<string, string | null>();
  const add = (name: string, parent: string | null, x: number, y: number, z = 0) => {
    positions.set(name, new Vector3(x, y, z));
    parentOf.set(name, parent);
  };

  add('hips', null, 0, -1.3);
  add('spine', 'hips', 0, -0.95);
  add('chest', 'spine', 0, -0.5);
  add('neck', 'chest', 0, 0);
  add('head', 'neck', 0, 0.16);

  const arm = {
    left: 1,
    right: -1,
  } as const;

  for (const side of SIDES) {
    const s = arm[side];
    add(`${side}Shoulder`, 'chest', 0.1 * s, -0.02);
    add(`${side}UpperArm`, `${side}Shoulder`, 0.5 * s, -0.06);
    add(`${side}LowerArm`, `${side}UpperArm`, 1.12 * s, -0.06);
    add(`${side}Hand`, `${side}LowerArm`, 1.66 * s, -0.06);

    const fingerBase: Array<[Finger, number, number, number]> = [
      // x offset from the hand bone, z lateral offset, first segment length
      ['Thumb', 0.06, 0.06, 0.085],
      ['Index', 0.15, -0.045, 0.09],
      ['Middle', 0.15, -0.012, 0.10],
      ['Ring', 0.15, 0.015, 0.092],
      ['Little', 0.15, 0.045, 0.075],
    ];
    for (const [finger, dx, dz, l1] of fingerBase) {
      const bones = fingerBoneNames(side, finger);
      const lengths = finger === 'Thumb' ? [l1, 0.06, 0.05] : [l1, l1 * 0.66, l1 * 0.52];
      let x = 1.66 * s + dx * s;
      const z = dz * s;
      let parent = `${side}Hand`;
      bones.forEach((bone, i) => {
        if (i > 0) x += lengths[i - 1] * s;
        add(bone, parent, x, -0.06, z);
        parent = bone;
      });
    }
  }

  return buildSkeleton(
    positions,
    parentOf,
    'leftUpperArm',
    'rightUpperArm',
    'neck',
    'default',
  );
}

/** Rest quaternion for every joint in normalized VRM space (usually identity). */
export function defaultWorldQuaternions(skeleton: Skeleton): Map<string, Quaternion> {
  const map = new Map<string, Quaternion>();
  for (const name of skeleton.order) map.set(name, new Quaternion());
  return map;
}
