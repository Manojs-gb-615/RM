import { describe, expect, it } from 'vitest';
import { createConfig } from '../src/avatar';
import { extractFaceChannels, NeutralCalibrator } from '../src/avatar/faceChannels';
import { NUM_POINTS } from '../src/avatar/layout';
import { readNpy } from '../eval/npy';
import path from 'node:path';

const demoDir = path.join(process.cwd(), '..', 'demo_data');
const session = readNpy(path.join(demoDir, 'synthetic_session.npy'));

describe('filtering', () => {
  it('reduces high-frequency noise without large lag', async () => {
    const { filterSequence } = await import('../src/avatar/filters');
    const T = 60;
    const frames = new Float32Array(T * NUM_POINTS * 2);
    for (let t = 0; t < T; t++) {
      for (let p = 0; p < NUM_POINTS; p++) {
        const base = 0.5 * Math.sin((t / 24) * Math.PI * 2);
        const noise = (p % 2 ? -1 : 1) * 0.004;
        frames[(t * NUM_POINTS + p) * 2] = base + noise;
        frames[(t * NUM_POINTS + p) * 2 + 1] = -base + noise;
      }
    }
    const config = createConfig();
    config.filter.body = { minCutoff: 0.5, beta: 0, dCutoff: 1 };
    const filtered = filterSequence(frames, 24, config);
    const energy = (data: Float32Array) => {
      let total = 0;
      for (let t = 2; t < T; t++) {
        const o = (t * NUM_POINTS + 7) * 2;
        const prev = ((t - 1) * NUM_POINTS + 7) * 2;
        const prev2 = ((t - 2) * NUM_POINTS + 7) * 2;
        total += Math.abs(data[o] - 2 * data[prev] + data[prev2]);
      }
      return total;
    };
    expect(energy(filtered.frames)).toBeLessThan(energy(frames) * 0.7);
  });

  it('rejects implausible joint speed spikes', async () => {
    const { filterSequence } = await import('../src/avatar/filters');
    const T = 10;
    const frames = new Float32Array(T * NUM_POINTS * 2);
    frames[(4 * NUM_POINTS + 7) * 2] = 3.0; // teleport spike for the wrist
    const result = filterSequence(frames, 24, createConfig());
    expect(result.outliers[4]).toBeGreaterThan(0);
  });
});

describe('face channels', () => {
  it('reports plausible values on synthetic face data', () => {
    const frame = new Float32Array(session.data.slice(0, NUM_POINTS * 2));
    const channels = extractFaceChannels(frame);
    for (const value of Object.values(channels)) expect(Number.isFinite(value)).toBe(true);
    expect(channels.eyeOpen).toBeGreaterThan(0);
    expect(channels.browRaise).toBeGreaterThan(0);
    expect(Math.abs(channels.gazeX)).toBeLessThan(1.5);
  });

  it('neutral calibration gives zero brow deviation for a constant face', () => {
    const frame = new Float32Array(session.data.slice(0, NUM_POINTS * 2));
    const raw = extractFaceChannels(frame);
    const calibrator = new NeutralCalibrator(5);
    for (let i = 0; i < 5; i++) calibrator.add(raw);
    const calibrated = calibrator.apply(raw);
    expect(calibrated.browRaise).toBeCloseTo(0, 6);
    expect(calibrated.eyeOpen).toBeCloseTo(1, 6);
  });
});
