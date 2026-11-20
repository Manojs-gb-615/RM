import React, { useState, useRef, useEffect } from 'react';
import { Upload, Video, FileText, ArrowRightLeft, Info } from 'lucide-react';
import PoseAnimationViewer from '../components/PoseAnimationViewer';
import AnimationControls from '../components/AnimationControls';
import StatusBadge from '../components/StatusBadge';
import { api } from '../api';
import type { TranscriptSegment, RetrievedSegment, StatusResponse, RetrievalResult } from '../api';

type Mode = 'text_only' | 'context_aware';

interface PoseSession {
  segments: RetrievedSegment[];
  mode: Mode;
}

export default function StudioPage() {
  // Backend status
  const [status, setStatus] = useState<StatusResponse | null>(null);

  // Video
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  // Transcript
  const [transcriptFile, setTranscriptFile] = useState<File | null>(null);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [activeSegIdx, setActiveSegIdx] = useState<number>(-1);

  // Pose session
  const [session, setSession] = useState<PoseSession | null>(null);
  const [mode, setMode] = useState<Mode>('context_aware');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Animation controls
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [externalFrame, setExternalFrame] = useState<number | undefined>(undefined);

  // Compare mode
  const [showCompare, setShowCompare] = useState(false);
  const [compareResult, setCompareResult] = useState<{ text_only: RetrievalResult; context_aware: RetrievalResult } | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  // Fetch backend status on mount
  useEffect(() => {
    api.status().then(setStatus).catch(() => setStatus(null));
  }, []);

  // Sync active segment to video time
  useEffect(() => {
    const video = videoRef.current;
    if (!video || segments.length === 0) return;
    const onTimeUpdate = () => {
      const t = video.currentTime;
      const idx = segments.findIndex(s => t >= s.start_seconds && t < s.end_seconds);
      if (idx !== -1 && idx !== activeSegIdx) {
        setActiveSegIdx(idx);
        setCurrentFrame(0);
        setExternalFrame(0);
      }
    };
    video.addEventListener('timeupdate', onTimeUpdate);
    return () => video.removeEventListener('timeupdate', onTimeUpdate);
  }, [segments, activeSegIdx]);

  // ── Handlers ────────────────────────────────────────────────────────────────

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setSession(null);
    setSegments([]);
    setActiveSegIdx(-1);
    setError(null);
  };

  const handleTranscriptUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setTranscriptFile(file);
    setError(null);
    try {
      const parsed = await api.parseTranscript(file);
      setSegments(parsed.segments);
    } catch (err: any) {
      setError(`Transcript parse error: ${err.message}`);
    }
  };

  const handleGenerate = async () => {
    if (!transcriptFile || segments.length === 0) {
      setError('Upload a transcript first.');
      return;
    }
    setLoading(true);
    setError(null);
    setSession(null);
    try {
      const result = await api.retrieveAll(transcriptFile, mode);
      setSession({ segments: result.results, mode });
      setActiveSegIdx(0);
      setCurrentFrame(0);
    } catch (err: any) {
      setError(`Generation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (activeSegIdx < 0 || segments.length === 0) return;
    const seg = segments[activeSegIdx];
    const prevSeg = activeSegIdx > 0 ? segments[activeSegIdx - 1] : null;
    const prevPose = session && activeSegIdx > 0
      ? session.segments[activeSegIdx - 1].retrieval.pose
      : null;

    setCompareLoading(true);
    try {
      const r = await api.compare({
        current_sentence: seg.text,
        prev_sentence: prevSeg?.text,
        prev_pose: prevPose ?? undefined,
      });
      setCompareResult({ text_only: r.text_only, context_aware: r.context_aware });
      setShowCompare(true);
    } catch (err: any) {
      setError(`Compare failed: ${err.message}`);
    } finally {
      setCompareLoading(false);
    }
  };

  // Current pose for active segment
  const activePose = session && activeSegIdx >= 0
    ? session.segments[activeSegIdx]?.retrieval.pose ?? null
    : null;
  const activeResult = session && activeSegIdx >= 0
    ? session.segments[activeSegIdx]?.retrieval
    : null;

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            SignLingo AI Mini
          </span>
          <span className="text-xs text-gray-500">Context-Aware ASL Pose Retrieval</span>
        </div>
        <StatusBadge status={status} />
      </header>

      {/* Research disclaimer */}
      <div className="bg-yellow-950 border-b border-yellow-800 px-6 py-2 flex items-center gap-2 text-yellow-300 text-xs">
        <Info size={14} />
        <span>
          <strong>Research prototype.</strong> Retrieved poses are experimental and have not been evaluated by qualified ASL signers.
          Poses represent <em>nearest-match retrieval</em> from a small dataset — not certified ASL translation.
        </span>
      </div>

      {/* Main content */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 max-w-7xl mx-auto w-full">

        {/* LEFT — Video + controls */}
        <div className="flex flex-col gap-4">
          <div className="card flex flex-col gap-3">
            <h2 className="font-semibold text-gray-200 flex items-center gap-2"><Video size={16} /> Educational Video</h2>

            <label className="btn-secondary cursor-pointer flex items-center gap-2 justify-center">
              <Upload size={15} /> Upload MP4
              <input type="file" accept="video/mp4,video/*" className="hidden" onChange={handleVideoUpload} />
            </label>

            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                className="rounded-lg w-full border border-gray-700"
                controls
                style={{ maxHeight: 320 }}
              />
            ) : (
              <div className="rounded-lg border-2 border-dashed border-gray-700 flex items-center justify-center h-48 text-gray-500 text-sm">
                No video uploaded
              </div>
            )}
          </div>

          {/* Transcript upload + generate */}
          <div className="card flex flex-col gap-3">
            <h2 className="font-semibold text-gray-200 flex items-center gap-2"><FileText size={16} /> Transcript</h2>
            <label className="btn-secondary cursor-pointer flex items-center gap-2 justify-center">
              <Upload size={15} /> Upload .srt / .vtt
              <input type="file" accept=".srt,.vtt" className="hidden" onChange={handleTranscriptUpload} />
            </label>
            {segments.length > 0 && (
              <p className="text-xs text-green-400">{segments.length} segments loaded.</p>
            )}

            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400">Mode:</label>
              <select
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm flex-1"
                value={mode}
                onChange={e => setMode(e.target.value as Mode)}
              >
                <option value="context_aware">Context-Aware</option>
                <option value="text_only">Text-Only Baseline</option>
              </select>
            </div>

            <button
              className="btn-primary flex items-center gap-2 justify-center"
              onClick={handleGenerate}
              disabled={loading || segments.length === 0}
            >
              {loading ? 'Retrieving…' : 'Generate ASL Poses'}
            </button>

            {error && (
              <div className="text-red-400 text-sm bg-red-950 border border-red-800 rounded p-2">{error}</div>
            )}
          </div>

          {/* Transcript segments list */}
          {segments.length > 0 && (
            <div className="card flex flex-col gap-2 max-h-64 overflow-y-auto">
              <h2 className="font-semibold text-gray-200 text-sm">Transcript</h2>
              {segments.map((seg, i) => (
                <button
                  key={seg.index}
                  onClick={() => { setActiveSegIdx(i); setCurrentFrame(0); }}
                  className={`text-left rounded px-2 py-1 text-sm transition-colors ${
                    i === activeSegIdx ? 'bg-primary/20 text-primary border border-primary/40' : 'hover:bg-gray-800 text-gray-300'
                  }`}
                >
                  <span className="text-xs text-gray-500 mr-2">
                    {seg.start_seconds.toFixed(1)}s
                  </span>
                  {seg.text}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* RIGHT — Pose viewer */}
        <div className="flex flex-col gap-4">
          <div className="card flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-gray-200">ASL Pose Viewer</h2>
              {session && (
                <button
                  className="btn-secondary text-xs flex items-center gap-1"
                  onClick={handleCompare}
                  disabled={compareLoading}
                >
                  <ArrowRightLeft size={13} />
                  {compareLoading ? 'Comparing…' : 'Compare modes'}
                </button>
              )}
            </div>

            {/* Active sentence display */}
            {activeSegIdx >= 0 && segments[activeSegIdx] && (
              <div className="bg-gray-800 rounded p-2 text-sm text-gray-200 border border-gray-700">
                <span className="text-xs text-gray-500 mr-1">{segments[activeSegIdx].start_seconds.toFixed(1)}s –</span>
                {segments[activeSegIdx].text}
              </div>
            )}

            {/* No-match notice */}
            {activeResult?.no_match_reason && (
              <div className="text-yellow-300 text-xs bg-yellow-950 border border-yellow-800 rounded p-2 flex items-start gap-1">
                <Info size={13} className="mt-0.5 shrink-0" />
                {activeResult.no_match_reason}
              </div>
            )}

            <div className="flex justify-center">
              <PoseAnimationViewer
                poses={activePose}
                label={session ? `${session.mode} · score ${activeResult?.score.toFixed(2) ?? '—'}` : 'no session'}
                labelColor={mode === 'context_aware' ? '#6366f1' : '#06b6d4'}
                isPlaying={isPlaying}
                playbackSpeed={speed}
                showLandmarks={showLandmarks}
                width={380}
                height={380}
                onFrameChange={(f, _t) => setCurrentFrame(f)}
                externalFrame={externalFrame}
              />
            </div>

            <AnimationControls
              isPlaying={isPlaying}
              onPlayPause={() => setIsPlaying(p => !p)}
              onRestart={() => { setIsPlaying(false); setCurrentFrame(0); setExternalFrame(0); }}
              speed={speed}
              onSpeedChange={setSpeed}
              currentFrame={currentFrame}
              totalFrames={activePose?.length ?? 0}
              onScrub={f => { setExternalFrame(f); setCurrentFrame(f); }}
              showLandmarks={showLandmarks}
              onToggleLandmarks={() => setShowLandmarks(p => !p)}
            />
          </div>

          {/* Match info */}
          {activeResult && (
            <div className="card text-xs text-gray-400 flex flex-col gap-1">
              <div className="font-semibold text-gray-300 text-sm">Retrieval Info</div>
              <div>Mode: <span className="text-gray-200">{activeResult.mode}</span></div>
              <div>Score: <span className="text-gray-200">{activeResult.score.toFixed(3)}</span></div>
              {activeResult.matched_sentence && (
                <div>Matched: <span className="text-gray-200 italic">"{activeResult.matched_sentence}"</span></div>
              )}
              {activeResult.matched_sample_id && (
                <div>Sample ID: <code className="text-gray-300">{activeResult.matched_sample_id}</code></div>
              )}
              {!activeResult.pose && (
                <div className="text-yellow-400">⚠ Caption-only fallback — no pose retrieved.</div>
              )}
            </div>
          )}

          {/* Compare panel */}
          {showCompare && compareResult && (
            <div className="card flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-sm">Mode Comparison</h3>
                <button className="text-xs text-gray-500 hover:text-gray-300" onClick={() => setShowCompare(false)}>✕</button>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {(['text_only', 'context_aware'] as const).map(m => {
                  const r = compareResult[m];
                  return (
                    <div key={m} className="flex flex-col gap-2">
                      <div className="text-xs font-semibold text-gray-400 uppercase">{m.replace('_', ' ')}</div>
                      <PoseAnimationViewer
                        poses={r.pose}
                        label={m === 'text_only' ? 'text-only' : 'context-aware'}
                        labelColor={m === 'text_only' ? '#06b6d4' : '#6366f1'}
                        isPlaying={false}
                        showLandmarks={false}
                        width={180}
                        height={180}
                      />
                      <div className="text-xs text-gray-400">
                        Score: {r.score.toFixed(3)}<br />
                        {r.matched_sentence
                          ? <span className="italic">"{r.matched_sentence.slice(0, 40)}…"</span>
                          : <span className="text-yellow-400">{r.no_match_reason}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
              {compareResult.text_only.matched_sample_id === compareResult.context_aware.matched_sample_id
                ? <p className="text-xs text-gray-500">Both modes matched the same clip.</p>
                : <p className="text-xs text-green-400">Context changed the retrieval result ✓</p>}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
