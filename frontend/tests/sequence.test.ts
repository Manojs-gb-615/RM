import { describe, expect, it } from 'vitest';
import { concatSequences, Playback, type PoseSequenceData } from '../src/avatar';
import { NUM_POINTS } from '../src/avatar/layout';

function clip(id: string, frameCount: number, startSentence: number): PoseSequenceData {
  const frames = new Float32Array(frameCount * NUM_POINTS * 2);
  for (let t = 0; t < frameCount; t++) frames[t * NUM_POINTS * 2] = t;
  return {
    id,
    title: id,
    fps: 24,
    frames,
    frameCount,
    sentences: [
      { index: startSentence, startFrame: 0, endFrame: frameCount, text: id },
    ],
    source: 'synthetic',
    synthetic: true,
    warning: '',
    text: id,
    layout: 'openpose137',
    normalization: 'neck_origin_shoulder_width',
  };
}

describe('sequence composition', () => {
  it('removes overlap frames and offsets sentence boundaries', () => {
    const a = clip('a', 100, 0);
    const b = clip('b', 50, 1);
    const session = concatSequences([a, b], 7);
    expect(session.frameCount).toBe(143);
    expect(session.sentences).toHaveLength(2);
    expect(session.sentences[1].startFrame).toBe(93);
    expect(session.sentences[1].index).toBe(1);
  });

  it('crossfades the transition frames instead of jumping', () => {
    const a = clip('a', 100, 0);
    const b = clip('b', 50, 1);
    const session = concatSequences([a, b], 7);
    // Frame 93 is the first blended frame: between a's 93 and b's 0.
    const value = session.frames[93 * NUM_POINTS * 2];
    expect(value).toBeGreaterThan(0);
    expect(value).toBeLessThan(93);
    // The blend approaches the next clip monotonically.
    const last = session.frames[99 * NUM_POINTS * 2];
    expect(last).toBeLessThan(value);
  });
});

describe('playback', () => {
  it('uses the clip fps and wraps when looping', () => {
    const playback = new Playback(clip('a', 48, 0));
    expect(playback.duration).toBeCloseTo(47 / 24, 6);
    playback.playing = true;
    playback.loop = true;
    playback.tick(2.0);
    expect(playback.time).toBeLessThan(playback.duration);
    playback.seekFrame(24);
    expect(playback.currentFrame).toBe(24);
    expect(playback.sentenceAt()?.text).toBe('a');
  });
});
