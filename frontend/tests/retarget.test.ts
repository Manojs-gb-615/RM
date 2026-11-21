import { describe, expect, it } from 'vitest';
import { Quaternion, Vector3 } from 'three';
import { createConfig, forwardKinematics, Retargeter, createDefaultSkeleton } from '../src/avatar';
import { NUM_POINTS } from '../src/avatar/layout';
import { readNpy } from '../eval/npy';
import path from 'node:path';

const demoDir = path.join(process.cwd(), '..', 'demo_data');
const session = readNpy(path.join(demoDir, 'synthetic_session.npy'));
const frames = session.data;

function frameAt(index: number): Float32Array {
  return frames.slice(index * NUM_POINTS * 2, (index + 1) * NUM_POINTS * 2);
}

describe('retargeting on synthetic ground truth', () => {
  const skeleton = createDefaultSkeleton();
  const config = createConfig();
  const retargeter = new Retargeter(skeleton, config);

  it('produces unit quaternions for every bone', () => {
    const pose = retargeter.solve(frameAt(0), 1 / 24);
    expect(pose.locals.size).toBe(skeleton.order.length);
    for (const q of pose.locals.values()) {
      expect(q).toBeInstanceOf(Quaternion);
      expect(q.length()).toBeCloseTo(1, 5);
    }
  });

  it('keeps FK bone lengths constant over the clip', () => {
    const retargeter = new Retargeter(skeleton, config);
    const lengths = new Map<string, number[]>();
    for (let t = 0; t < Math.min(80, frames.length / (NUM_POINTS * 2)); t += 3) {
      const pose = retargeter.solve(frameAt(t), 1 / 24);
      const { positions } = forwardKinematics(skeleton, pose.locals);
      for (const name of skeleton.order) {
        const joint = skeleton.joints.get(name)!;
        if (!joint.parent) continue;
        const len = positions.get(name)!.distanceTo(positions.get(joint.parent)!);
        const list = lengths.get(name) ?? [];
        list.push(len);
        lengths.set(name, list);
      }
    }
    for (const [name, values] of lengths) {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      if (mean < 1e-4) continue;
      const max = Math.max(...values);
      const min = Math.min(...values);
      expect((max - min) / mean, name).toBeLessThan(1e-4);
    }
  });

  it('places the avatar hand at the observed wrist', () => {
    const retargeter = new Retargeter(skeleton, config);
    for (let t = 0; t < 60; t++) {
      const frame = frameAt(t);
      const pose = retargeter.solve(frame, 1 / 24);
      if (t % 10 !== 0) continue;
      const { positions } = forwardKinematics(skeleton, pose.locals);
      const leftWrist = new Vector3(frame[7 * 2], frame[7 * 2 + 1], 0);
      const rightWrist = new Vector3(frame[4 * 2], frame[4 * 2 + 1], 0);
      const leftHand = positions.get('leftHand')!;
      const rightHand = positions.get('rightHand')!;
      expect(Math.hypot(leftHand.x - leftWrist.x, leftHand.y - leftWrist.y)).toBeLessThan(0.3);
      expect(Math.hypot(rightHand.x - rightWrist.x, rightHand.y - rightWrist.y)).toBeLessThan(0.3);
    }
  });

  it('produces finite face channels and eyes above the mouth', () => {
    const retargeter = new Retargeter(skeleton, config);
    for (let t = 0; t < 50; t += 5) {
      const pose = retargeter.solve(frameAt(t), 1 / 24);
      for (const value of Object.values(pose.face)) expect(Number.isFinite(value)).toBe(true);
      expect(pose.face.eyeOpen).toBeGreaterThan(0);
    }
  });

  it('is deterministic for the same input and reset', () => {
    const a = new Retargeter(skeleton, config);
    const b = new Retargeter(skeleton, config);
    for (let t = 0; t < 12; t++) {
      a.solve(frameAt(t), 1 / 24);
      b.solve(frameAt(t), 1 / 24);
    }
    const pa = a.solve(frameAt(12), 1 / 24);
    const pb = b.solve(frameAt(12), 1 / 24);
    for (const [name, q] of pa.locals) {
      expect(q.angleTo(pb.locals.get(name)!)).toBeLessThan(1e-6);
    }
  });
});
