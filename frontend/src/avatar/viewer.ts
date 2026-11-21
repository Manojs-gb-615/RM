import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  HemisphereLight,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Scene,
  Vector3,
  WebGLRenderer,
} from 'three';
import { fetchSequence } from './api';
import type { AvatarConfig } from './config';
import type { FaceCapabilities } from './faceRig';
import { filterSequence } from './filters';
import { NUM_POINTS } from './layout';
import { MannequinAvatar } from './mannequin';
import { Playback } from './playback';
import { Retargeter } from './retarget';
import { lerpFrames, type PoseSequenceData } from './sequence';
import { StickFigure } from './stickFigure';
import { VrmAvatar } from './vrmAvatar';

export interface ViewerStats {
  fps: number;
  retargetMs: number;
  frame: number;
  frameCount: number;
  sentence: string;
  source: 'real' | 'synthetic' | string;
  synthetic: boolean;
  depthFlips: number;
  handConfidence: number;
  backends: string;
}

export type AvatarInstance = VrmAvatar | MannequinAvatar;

export interface ViewerOptions {
  canvas: HTMLCanvasElement;
  stickCanvas: HTMLCanvasElement;
  config: AvatarConfig;
  onStats?: (stats: ViewerStats) => void;
}

/**
 * Demo viewer: owns the Three.js scene, playback clock, browser-side
 * retargeting and the side-by-side stick figure.
 */
export class AvatarViewer {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(32, 1, 0.01, 100);
  private readonly renderer: WebGLRenderer;
  private readonly stick: StickFigure;
  private readonly playback = new Playback({
    id: '',
    title: '',
    fps: 24,
    frames: new Float32Array(0),
    frameCount: 0,
    sentences: [],
    source: 'synthetic',
    synthetic: true,
    warning: '',
    text: '',
    layout: 'openpose137',
    normalization: 'neck_origin_shoulder_width',
  });
  private clip: PoseSequenceData | null = null;
  private filtered: Float32Array = new Float32Array(0);
  private avatar: AvatarInstance | null = null;
  private retargeter: Retargeter | null = null;
  private readonly outline: Points;
  private readonly outlineGeometry: BufferGeometry;
  private readonly scratch = new Float32Array(NUM_POINTS * 2);
  private stats: ViewerStats = emptyStats();
  private lastTime = 0;
  private raf = 0;
  private running = false;
  private frameAccumulator = 0;
  private frameCounter = 0;
  private recorder: MediaRecorder | null = null;
  private recordedChunks: BlobPart[] = [];
  private resizeObserver: ResizeObserver | null = null;
  private cameraSide = 1;

  constructor(private options: ViewerOptions) {
    this.renderer = new WebGLRenderer({ canvas: options.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(new Color(options.config.render.background), 1);
    this.scene.background = new Color(options.config.render.background);

    const hemi = new HemisphereLight(0xffffff, 0x303844, 2.2);
    this.scene.add(hemi);
    const key = new DirectionalLight(0xffffff, 2.0);
    key.position.set(2, 4, 3);
    this.scene.add(key);
    const fill = new DirectionalLight(0x99bbff, 0.8);
    fill.position.set(-3, 1, -2);
    this.scene.add(fill);

    this.stick = new StickFigure(options.stickCanvas);
    this.outlineGeometry = new BufferGeometry();
    this.outlineGeometry.setAttribute(
      'position',
      new BufferAttribute(new Float32Array(NUM_POINTS * 3), 3),
    );
    this.outline = new Points(
      this.outlineGeometry,
      new PointsMaterial({
        color: 0x7dd3fc,
        size: 0.018,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
        blending: AdditiveBlending,
      }),
    );
    this.outline.visible = options.config.render.showKeypoints;
    this.scene.add(this.outline);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(options.canvas);
    this.resize();

    this.setAvatar(new MannequinAvatar());
    this.start();
  }

  // ---------------------------------------------------------------- avatar
  setAvatar(avatar: AvatarInstance): void {
    if (this.avatar) {
      this.scene.remove(this.avatar.scene);
      this.avatar.dispose();
    }
    this.avatar = avatar;
    this.scene.add(avatar.scene);
    this.retargeter = new Retargeter(avatar.skeleton, this.options.config);
    this.cameraSide = this.options.config.camera.autoFacing ? avatar.facingSign() : 1;
    avatar.setSkinTone(this.options.config.render.skinTone);
    this.updateCamera();
    if (this.clip) this.retargeter.reset();
  }

  async loadAvatarFile(file: File): Promise<AvatarInstance> {
    const url = URL.createObjectURL(file);
    try {
      const avatar = await VrmAvatar.load(url, this.options.config);
      this.setAvatar(avatar);
      return avatar;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  get currentAvatar(): AvatarInstance | null {
    return this.avatar;
  }

  get capabilities(): FaceCapabilities | null {
    return this.avatar instanceof VrmAvatar ? this.avatar.capabilities : null;
  }

  // -------------------------------------------------------------- sequence
  setSequence(clip: PoseSequenceData): void {
    this.clip = clip;
    this.playback.setClip(clip);
    this.playback.seekFrame(0);
    this.refilter();
  }

  async loadSequence(sequenceId: string, base?: string): Promise<PoseSequenceData> {
    const clip = await fetchSequence(sequenceId, base);
    this.setSequence(clip);
    return clip;
  }

  refilter(): void {
    if (!this.clip) return;
    const result = filterSequence(
      this.clip.frames,
      this.clip.fps,
      this.options.config,
    );
    this.filtered = result.frames;
    this.retargeter?.reset();
  }

  setConfig(config: AvatarConfig): void {
    this.options.config = config;
    this.retargeter?.setConfig(config);
    this.scene.background = new Color(config.render.background);
    this.renderer.setClearColor(new Color(config.render.background), 1);
    this.outline.visible = config.render.showKeypoints;
    this.stick.resize();
    if (config.camera.autoFacing && this.avatar) this.cameraSide = this.avatar.facingSign();
    else if (!config.camera.autoFacing) this.cameraSide = 1;
    this.updateCamera();
  }

  setSkinTone(color: string): boolean {
    this.options.config.render.skinTone = color;
    return this.avatar?.setSkinTone(color) ?? false;
  }

  // -------------------------------------------------------------- playback
  play(): void {
    this.playback.playing = true;
  }
  pause(): void {
    this.playback.playing = false;
  }
  toggle(): void {
    this.playback.playing = !this.playback.playing;
  }
  get playing(): boolean {
    return this.playback.playing;
  }
  seekFrame(frame: number): void {
    this.playback.seekFrame(frame);
    this.retargeter?.reset();
  }
  seekSeconds(seconds: number): void {
    this.playback.seekSeconds(seconds);
    this.retargeter?.reset();
  }
  setSpeed(speed: number): void {
    this.playback.speed = speed;
  }
  setLoop(loop: boolean): void {
    this.playback.loop = loop;
  }
  get currentTime(): number {
    return this.playback.time;
  }
  get duration(): number {
    return this.playback.duration;
  }
  get frameCount(): number {
    return this.playback.frameCount;
  }
  get sentenceText(): string {
    return this.playback.sentenceAt()?.text ?? '';
  }

  // --------------------------------------------------------------- render
  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastTime = performance.now();
    const loop = (now: number) => {
      if (!this.running) return;
      const dt = Math.min(0.1, Math.max(0, (now - this.lastTime) / 1000));
      this.lastTime = now;
      this.update(dt);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  private update(dt: number): void {
    this.playback.tick(dt);
    const sample = this.playback.sample();
    if (this.filtered.length > 0) {
      lerpFrames(this.filtered, sample.frameA, sample.frameB, sample.alpha, this.scratch);
      if (this.retargeter && this.avatar) {
        const started = performance.now();
        const pose = this.retargeter.solve(this.scratch, dt);
        this.avatar.apply(pose);
        this.avatar.update(dt);
        this.updateOutline();
        this.stats.retargetMs = performance.now() - started;
        this.stats.depthFlips = pose.meta.depthFlips;
        this.stats.handConfidence = pose.meta.handConfidence;
        this.stats.frame = sample.frameA;
        this.stats.frameCount = this.playback.frameCount;
        this.stats.sentence = this.sentenceText;
        this.stats.source = this.clip?.source ?? 'synthetic';
        this.stats.synthetic = this.clip?.synthetic ?? true;
        this.stats.backends =
          this.avatar instanceof VrmAvatar
            ? `VRM ${this.avatar.version}`
            : 'procedural mannequin';
      }
      this.stick.draw(this.scratch, {
        background: this.options.config.render.background,
        mirror: this.options.config.mirror,
        ...(this.clip?.synthetic ? { left: '#4ec9b0' } : {}),
      });
    }

    this.renderer.render(this.scene, this.camera);
    this.frameAccumulator += dt;
    this.frameCounter++;
    if (this.frameAccumulator >= 0.5) {
      this.stats.fps = this.frameCounter / this.frameAccumulator;
      this.frameAccumulator = 0;
      this.frameCounter = 0;
      this.options.onStats?.({ ...this.stats });
    }
  }

  private updateOutline(): void {
    if (!this.outline.visible || !this.avatar) return;
    const attribute = this.outlineGeometry.getAttribute('position') as BufferAttribute;
    const unit = this.avatar.unitScale;
    const facing = this.avatar.facingSign();
    const neck = this.avatar.neckWorld;
    for (let i = 0; i < NUM_POINTS; i++) {
      const x = this.scratch[i * 2] * facing;
      const y = this.scratch[i * 2 + 1];
      attribute.setXYZ(i, neck.x + x * unit, neck.y + y * unit, neck.z + facing * 0.02);
    }
    attribute.needsUpdate = true;
    this.outlineGeometry.computeBoundingSphere();
  }

  private updateCamera(): void {
    const unit = this.avatar?.unitScale ?? 0.42;
    const neck = this.avatar?.neckWorld ?? new Vector3(0, 1.35, 0);
    const cfg = this.options.config.camera;
    const target = new Vector3(neck.x, neck.y + cfg.targetY * unit, neck.z);
    const distance = cfg.distance * unit;
    let yaw = cfg.autoFacing ? 0 : (cfg.yawDeg * Math.PI) / 180;
    if (cfg.autoFacing) yaw = 0;
    const side = this.cameraSide;
    this.camera.position.set(
      Math.sin(yaw) * distance,
      target.y,
      side * Math.cos(yaw) * distance,
    );
    this.camera.fov = cfg.fov;
    this.camera.near = Math.max(0.005, distance * 0.01);
    this.camera.far = distance * 10;
    this.camera.lookAt(target);
    this.camera.updateProjectionMatrix();
  }

  resize(): void {
    const canvas = this.renderer.domElement;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.stick.resize();
    this.updateCamera();
  }

  // -------------------------------------------------------------- recording
  startRecording(fps = 60): boolean {
    if (typeof MediaRecorder === 'undefined') return false;
    const stream = this.renderer.domElement.captureStream(fps);
    const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mimeType = types.find((type) => MediaRecorder.isTypeSupported(type));
    if (!mimeType) return false;
    this.recordedChunks = [];
    this.recorder = new MediaRecorder(stream, { mimeType });
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.recordedChunks.push(event.data);
    };
    this.recorder.start(250);
    return true;
  }

  stopRecording(): Promise<Blob | null> {
    const recorder = this.recorder;
    if (!recorder) return Promise.resolve(null);
    return new Promise((resolve) => {
      recorder.onstop = () => {
        this.recorder = null;
        resolve(new Blob(this.recordedChunks, { type: recorder.mimeType }));
      };
      recorder.stop();
    });
  }

  /** Debug snapshot of solver state, for the panel. */
  debugInfo(): Record<string, number | string> {
    const state = this.retargeter?.state;
    return {
      fps: this.stats.fps,
      retargetMs: this.stats.retargetMs,
      frame: this.playback.currentFrame,
      duration: this.playback.duration,
      flips: state?.flips ?? 0,
      clamped: state?.elbowClamped ?? 0,
      headResidual: state?.headResidual ?? 0,
      handConfidence: state?.handConfidence ?? 1,
      avatar: this.stats.backends,
    };
  }

  dispose(): void {
    this.stop();
    this.resizeObserver?.disconnect();
    this.avatar?.dispose();
    this.avatar = null;
    this.outlineGeometry.dispose();
    (this.outline.material as PointsMaterial).dispose();
    this.renderer.dispose();
  }
}

function emptyStats(): ViewerStats {
  return {
    fps: 0,
    retargetMs: 0,
    frame: 0,
    frameCount: 0,
    sentence: '',
    source: 'synthetic',
    synthetic: true,
    depthFlips: 0,
    handConfidence: 1,
    backends: 'procedural mannequin',
  };
}