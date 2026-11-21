import { describe, expect, it } from 'vitest';
import { clampElbowFlexion, solveArm, solveDepth } from '../src/avatar/depth';
import { createConfig } from '../src/avatar';
import { Vector3 } from 'three';

describe('weak-perspective depth', () => {
  it('solves |dz| from bone length and 2D displacement', () => {
    const result = solveDepth(0.6, 0, 1.0, 0, 0.25, 0.35);
    expect(result.dz).toBeCloseTo(0.8, 6);
    expect(result.clamped).toBe(false);
  });

  it('keeps the previous sign when the alternative is only marginally better', () => {
    const forward = solveDepth(0.1, 0, 1.0, 0.9, 0, 0.35);
    const flipped = solveDepth(0.1, 0, 1.0, -0.9, 0, 0.35);
    expect(Math.sign(forward.dz)).toBe(1);
    expect(Math.sign(flipped.dz)).toBe(-1);
  });

  it('clamps when 2D noise exceeds the bone length', () => {
    const result = solveDepth(2, 0, 1, 0, 0, 0.35);
    expect(result.clamped).toBe(true);
    expect(Math.abs(result.dz)).toBeLessThan(0.02);
  });
});

describe('elbow limits', () => {
  it('does not bend beyond the configured range', () => {
    const upper = new Vector3(0, 1, 0);
    const natural = new Vector3(Math.sin(1), Math.cos(1), 0);
    const folded = new Vector3(0, 1e-4, 0).normalize();
    expect(clampElbowFlexion(upper, natural, 0.1, 3.0).clamped).toBe(false);
    const clamped = clampElbowFlexion(upper, folded, 0.5, 3.0);
    expect(clamped.clamped).toBe(true);
    const angle = Math.acos(Math.min(1, upper.dot(clamped.dir)));
    expect(angle).toBeGreaterThanOrEqual(0.49);
  });

  it('keeps arm bone lengths after solving', () => {
    const config = createConfig();
    const shoulder = new Vector3(0.5, -0.06, 0);
    const solution = solveArm(
      shoulder,
      { x: 1.0, y: -0.7 },
      { x: 1.05, y: -0.1 },
      { x: 0.5, y: -0.06 },
      0.62,
      0.55,
      { elbowZ: 0, wristZ: 0 },
      config,
    );
    expect(solution.elbow.distanceTo(shoulder)).toBeCloseTo(0.62, 5);
    expect(solution.wrist.distanceTo(solution.elbow)).toBeCloseTo(0.55, 5);
  });
});
