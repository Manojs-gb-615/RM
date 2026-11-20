// React imported via JSX transform
import { Play, Pause, RotateCcw } from 'lucide-react';

interface AnimationControlsProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  onRestart: () => void;
  speed: number;
  onSpeedChange: (s: number) => void;
  currentFrame: number;
  totalFrames: number;
  onScrub: (f: number) => void;
  showLandmarks: boolean;
  onToggleLandmarks: () => void;
}

export default function AnimationControls({
  isPlaying, onPlayPause, onRestart,
  speed, onSpeedChange,
  currentFrame, totalFrames, onScrub,
  showLandmarks, onToggleLandmarks,
}: AnimationControlsProps) {
  return (
    <div className="flex flex-col gap-2 mt-2">
      {/* Scrubber */}
      <input
        type="range" min={0} max={Math.max(0, totalFrames - 1)}
        value={currentFrame}
        onChange={e => onScrub(Number(e.target.value))}
        className="w-full accent-primary"
        disabled={totalFrames === 0}
      />
      {/* Buttons row */}
      <div className="flex items-center gap-2 flex-wrap">
        <button className="btn-secondary p-2" onClick={onPlayPause} disabled={totalFrames === 0}>
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button className="btn-secondary p-2" onClick={onRestart} disabled={totalFrames === 0}>
          <RotateCcw size={16} />
        </button>
        <select
          className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm"
          value={speed}
          onChange={e => onSpeedChange(Number(e.target.value))}
        >
          {[0.5, 1, 1.5, 2].map(s => (
            <option key={s} value={s}>{s}×</option>
          ))}
        </select>
        <label className="flex items-center gap-1 text-sm cursor-pointer select-none">
          <input
            type="checkbox" checked={showLandmarks}
            onChange={onToggleLandmarks}
            className="accent-primary"
          />
          Landmarks
        </label>
        <span className="ml-auto text-xs text-gray-400">
          {totalFrames > 0 ? `Frame ${currentFrame + 1}/${totalFrames}` : 'No pose'}
        </span>
      </div>
    </div>
  );
}
