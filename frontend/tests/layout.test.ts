import { describe, expect, it } from 'vitest';
import { BODY_CONNECTIONS, createConfig, HAND_CONNECTIONS, IDX, LAYOUT, NUM_POINTS } from '../src/avatar';

describe('canonical layout', () => {
  it('is 137 points with body, face, left hand, right hand order', () => {
    expect(NUM_POINTS).toBe(137);
    expect(LAYOUT.regions.body).toMatchObject({ start: 0, count: 25 });
    expect(LAYOUT.regions.face).toMatchObject({ start: 25, count: 70 });
    expect(LAYOUT.regions.leftHand).toMatchObject({ start: 95, count: 21 });
    expect(LAYOUT.regions.rightHand).toMatchObject({ start: 116, count: 21 });
    expect(LAYOUT.regions.face.names).toHaveLength(70);
  });

  it('exposes the anatomical indices used by the solver', () => {
    expect(IDX.neck).toBe(1);
    expect(IDX.rightShoulder).toBe(2);
    expect(IDX.leftShoulder).toBe(5);
    expect(IDX.rightWrist).toBe(4);
    expect(IDX.leftWrist).toBe(7);
    expect(IDX.rightHandStart).toBe(116);
    expect(IDX.leftHandStart).toBe(95);
    expect(BODY_CONNECTIONS).toContainEqual([2, 3]);
    expect(HAND_CONNECTIONS[0]).toEqual([0, 1]);
  });

  it('builds a config that keeps layout wiring intact', () => {
    const config = createConfig();
    expect(config.layout.totalPoints).toBe(137);
    expect(config.depth.maxElbowFlexion).toBeGreaterThan(90);
    expect(config.hands.palmSignLeft).toBe(1);
  });
});
