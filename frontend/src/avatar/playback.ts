import type { PoseSequenceData, SentenceBoundary } from './sequence';

export interface PlaybackSample {
  frameA: number;
  frameB: number;
  alpha: number;
  time: number;
}

/**
 * Timeline playback with fps taken from the data (never hardcoded), plus
 * scrubbing, speed, loop and sentence-boundary queries.
 */
export class Playback {
  time = 0;
  speed = 1;
  playing = false;
  loop = true;
  fps: number;
  frameCount: number;

  constructor(public clip: PoseSequenceData) {
    this.fps = clip.fps > 0 ? clip.fps : 24;
    this.frameCount = clip.frameCount;
  }

  get duration(): number {
    return Math.max(1 / this.fps, (this.frameCount - 1) / this.fps);
  }

  setClip(clip: PoseSequenceData): void {
    this.clip = clip;
    this.fps = clip.fps > 0 ? clip.fps : 24;
    this.frameCount = clip.frameCount;
    this.time = Math.min(this.time, this.duration);
  }

  tick(dt: number): void {
    if (!this.playing) return;
    this.time += dt * this.speed;
    if (this.time >= this.duration) {
      if (this.loop) this.time %= this.duration;
      else {
        this.time = this.duration;
        this.playing = false;
      }
    }
    if (this.time < 0) this.time = 0;
  }

  seekFrame(frame: number, fractional = 0): void {
    const f = Math.max(0, Math.min(this.frameCount - 1, frame)) + fractional;
    this.time = f / this.fps;
  }

  seekSeconds(seconds: number): void {
    this.time = Math.max(0, Math.min(this.duration, seconds));
  }

  sample(): PlaybackSample {
    const exact = this.time * this.fps;
    const frameA = Math.floor(exact);
    const frameB = Math.min(this.frameCount - 1, frameA + 1);
    return { frameA, frameB, alpha: exact - frameA, time: this.time };
  }

  get currentFrame(): number {
    return Math.floor(this.time * this.fps);
  }

  sentenceAt(frame = this.currentFrame): SentenceBoundary | null {
    for (const sentence of this.clip.sentences) {
      if (frame >= sentence.startFrame && frame < sentence.endFrame) return sentence;
    }
    return this.clip.sentences[this.clip.sentences.length - 1] ?? null;
  }

  /** Frames from the nearest sentence boundary, or Infinity when far away. */
  framesFromBoundary(frame = this.currentFrame): number {
    let best = Infinity;
    for (const sentence of this.clip.sentences) {
      best = Math.min(best, Math.abs(frame - sentence.startFrame), Math.abs(frame - sentence.endFrame));
    }
    return best;
  }
}
