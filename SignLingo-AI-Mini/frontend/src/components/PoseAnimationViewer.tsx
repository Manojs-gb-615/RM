import { useRef, useEffect, useCallback, useState } from 'react';

// MediaPipe upper-body keypoint connections (indices 0-24)
const BODY_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
  [9,10],[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
];

// Hand connections (21 landmarks per hand)
function handConns(base: number): [number, number][] {
  return [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
  ].map(([a, b]) => [base + a, base + b]);
}
const LHAND_CONNECTIONS = handConns(25);
const RHAND_CONNECTIONS = handConns(46);

const COLORS = {
  body:   '#06b6d4',
  lhand:  '#22c55e',
  rhand:  '#f97316',
  face:   '#a855f7',
  dot:    '#ffffff',
};

interface PoseAnimationViewerProps {
  /** pose data: array of frames, each frame = 274 numbers [x0,y0,x1,y1,...] */
  poses: number[][] | null;
  label?: string;
  labelColor?: string;
  isPlaying: boolean;
  playbackSpeed?: number;
  showLandmarks?: boolean;
  width?: number;
  height?: number;
  onFrameChange?: (frame: number, total: number) => void;
  externalFrame?: number;        // controlled from outside (scrubber)
}

function drawFrame(
  ctx: CanvasRenderingContext2D,
  frameData: number[],
  W: number,
  H: number,
  showLandmarks: boolean,
) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, W, H);

  if (!frameData || frameData.length < 274) return;

  // Parse keypoints
  const pts: [number, number][] = [];
  for (let i = 0; i < 137; i++) {
    pts.push([frameData[i * 2] * W, frameData[i * 2 + 1] * H]);
  }

  const drawLine = (a: number, b: number, color: string) => {
    const [x1, y1] = pts[a];
    const [x2, y2] = pts[b];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  };

  BODY_CONNECTIONS.forEach(([a, b]) => drawLine(a, b, COLORS.body));
  LHAND_CONNECTIONS.forEach(([a, b]) => drawLine(a, b, COLORS.lhand));
  RHAND_CONNECTIONS.forEach(([a, b]) => drawLine(a, b, COLORS.rhand));

  if (showLandmarks) {
    pts.forEach(([x, y], i) => {
      let color = COLORS.dot;
      if (i < 25)       color = COLORS.body;
      else if (i < 46)  color = COLORS.lhand;
      else if (i < 67)  color = COLORS.rhand;
      else               color = COLORS.face;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

export default function PoseAnimationViewer({
  poses,
  label = 'generated',
  labelColor = '#6366f1',
  isPlaying,
  playbackSpeed = 1,
  showLandmarks = true,
  width = 400,
  height = 400,
  onFrameChange,
  externalFrame,
}: PoseAnimationViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef  = useRef(0);
  const rafRef    = useRef<number>(0);
  const lastRef   = useRef<number>(0);
  const [currentFrame, setCurrentFrame] = useState(0);

  const totalFrames = poses ? poses.length : 0;
  const FPS = 25 * playbackSpeed;
  const MS_PER_FRAME = 1000 / FPS;

  // Draw a specific frame
  const renderFrame = useCallback((f: number) => {
    const canvas = canvasRef.current;
    if (!canvas || !poses) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    drawFrame(ctx, poses[f], width, height, showLandmarks);
    setCurrentFrame(f);
    onFrameChange?.(f, totalFrames);
  }, [poses, width, height, showLandmarks, totalFrames, onFrameChange]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying || !poses || poses.length === 0) {
      cancelAnimationFrame(rafRef.current);
      return;
    }
    lastRef.current = 0;

    const loop = (ts: number) => {
      if (!lastRef.current) lastRef.current = ts;
      const elapsed = ts - lastRef.current;
      if (elapsed >= MS_PER_FRAME) {
        lastRef.current = ts;
        frameRef.current = (frameRef.current + 1) % poses.length;
        renderFrame(frameRef.current);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying, poses, MS_PER_FRAME, renderFrame]);

  // External frame control (scrubber / pause)
  useEffect(() => {
    if (externalFrame !== undefined && !isPlaying && poses) {
      frameRef.current = externalFrame;
      renderFrame(externalFrame);
    }
  }, [externalFrame, isPlaying, renderFrame, poses]);

  // Initial render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (!poses || poses.length === 0) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#475569';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No pose data available', width / 2, height / 2 - 10);
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = '#334155';
      ctx.fillText('See README for dataset setup', width / 2, height / 2 + 15);
    } else {
      renderFrame(0);
    }
  }, [poses, width, height, renderFrame]);

  return (
    <div className="relative inline-block">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded-lg border border-gray-700"
        style={{ background: '#0f172a' }}
      />
      {/* Label badge */}
      <div
        className="absolute top-2 left-2 badge text-white text-xs"
        style={{ background: labelColor }}
      >
        {label}
      </div>
      {/* Frame counter */}
      {poses && poses.length > 0 && (
        <div className="absolute bottom-2 right-2 text-xs text-gray-400 bg-black/50 px-2 py-0.5 rounded">
          {currentFrame + 1} / {totalFrames}
        </div>
      )}
    </div>
  );
}
