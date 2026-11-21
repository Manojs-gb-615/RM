import { BODY_CONNECTIONS, HAND_CONNECTIONS, HAND_CONNECTIONS as _HC, IDX, NUM_POINTS } from './layout';

export interface StickStyle {
  background?: string;
  body: string;
  left: string;
  right: string;
  face: string;
  lineWidth: number;
  mirror: boolean;
}

export const DEFAULT_STICK_STYLE: StickStyle = {
  body: '#e8eef7',
  left: '#4ec9b0',
  right: '#e0a458',
  face: '#94a3b8',
  lineWidth: 2,
  mirror: false,
};

/**
 * 2D canvas stick figure using the canonical layout. Deliberately the same
 * geometry the avatar is driven from, so the side-by-side view is honest.
 */
export class StickFigure {
  private readonly ctx: CanvasRenderingContext2D;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas is unavailable');
    this.ctx = ctx;
  }

  resize(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw(frame: Float32Array, style: Partial<StickStyle> = {}): void {
    const s = { ...DEFAULT_STICK_STYLE, ...style };
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = s.background ?? 'transparent';
    ctx.fillRect(0, 0, w, h);

    const scale = Math.min(w / 3.4, h / 3.4);
    const originX = w / 2;
    const originY = h * 0.28;

    const px = (i: number) => originX + frame[i * 2] * scale * (s.mirror ? -1 : 1);
    const py = (i: number) => originY - frame[i * 2 + 1] * scale;

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = s.lineWidth;

    ctx.strokeStyle = s.body;
    for (const [a, b] of BODY_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }

    for (const [start, color] of [
      [IDX.leftHandStart, s.left],
      [IDX.rightHandStart, s.right],
    ] as const) {
      ctx.strokeStyle = color;
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(px(start + a), py(start + a));
        ctx.lineTo(px(start + b), py(start + b));
        ctx.stroke();
      }
    }

    ctx.fillStyle = s.face;
    for (let i = 25; i < 95; i += 2) {
      ctx.beginPath();
      ctx.arc(px(i), py(i), 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/** Kept as a named export for the viewer's legend. */
export const STICK_CONNECTIONS = BODY_CONNECTIONS;
export { _HC as HAND_LINKS, NUM_POINTS };
