import './style.css';
import {
  AvatarViewer,
  API_BASE,
  createConfig,
  fetchManifest,
  tryLoadDefaultAvatar,
  type AvatarConfig,
  type DemoManifest,
  type PoseSequenceData,
} from './avatar';

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="topbar">
    <div>
      <h1>ContextSign AI <span>· ASL Avatar Renderer</span></h1>
      <p class="subtitle">Ground-truth keypoints → browser retargeting → VRM. No model output, no signing accuracy claim.</p>
    </div>
    <div class="badges" id="badges"></div>
  </header>

  <div class="banner" id="banner" hidden></div>

  <main class="stage">
    <section class="panel">
      <div class="panel-head">
        <h2>Canonical keypoints</h2>
        <span class="hint">same data that drives the avatar</span>
      </div>
      <div class="canvas-wrap">
        <canvas id="stick"></canvas>
      </div>
      <div class="transcript">
        <div class="transcript-head">
          <span id="clip-title">No sequence loaded</span>
          <span id="sentence-index"></span>
        </div>
        <p id="sentence-text">Load a demo sequence from the FastAPI backend.</p>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Avatar</h2>
        <span class="hint" id="avatar-hint">procedural mannequin</span>
      </div>
      <div class="canvas-wrap avatar-wrap">
        <canvas id="avatar"></canvas>
        <div class="face-warning" id="face-warning" hidden></div>
      </div>
      <div class="face-channels" id="face-channels">
        <span>Neutral calibration starts on playback.</span>
      </div>
    </section>
  </main>

  <section class="controls">
    <div class="control-row">
      <label class="field">
        <span>Demo sequence</span>
        <select id="sequence"><option value="">Loading…</option></select>
      </label>
      <button id="play" class="primary" disabled>Play</button>
      <button id="restart" disabled>Restart</button>
      <div class="scrub">
        <input id="scrub" type="range" min="0" max="1000" value="0" disabled />
        <span id="time">0.00 / 0.00 s</span>
      </div>
      <label class="field inline">
        <span>Speed</span>
        <select id="speed">
          <option value="0.25">0.25×</option>
          <option value="0.5">0.5×</option>
          <option value="0.75">0.75×</option>
          <option value="1" selected>1×</option>
          <option value="1.5">1.5×</option>
          <option value="2">2×</option>
        </select>
      </label>
      <label class="check"><input id="loop" type="checkbox" checked /> Loop</label>
      <button id="record">Record WebM</button>
    </div>
    <div class="control-row secondary">
      <label class="field">
        <span>Avatar (.vrm)</span>
        <input id="avatar-file" type="file" accept=".vrm,model/gltf-binary" />
      </label>
      <label class="field inline">
        <span>Skin</span>
        <input id="skin" type="color" value="#c98d6b" />
      </label>
      <label class="field inline">
        <span>Background</span>
        <input id="background" type="color" value="#181c24" />
      </label>
      <label class="check"><input id="keypoints" type="checkbox" /> Keypoint overlay</label>
      <label class="check"><input id="mirror" type="checkbox" /> Mirror</label>
      <span class="status" id="status"></span>
    </div>
  </section>

  <section class="debug" id="debug">
    <button class="debug-toggle" id="debug-toggle">Debug panel ▾</button>
    <div class="debug-body" id="debug-body" hidden>
      <div class="debug-grid" id="debug-grid"></div>
      <pre id="debug-stats"></pre>
    </div>
  </section>

  <footer>
    <p><strong>Limitations:</strong> depth, palm orientation and elbow direction are inferred from 2D and are approximate.
    Fingerspelling clarity depends on avatar rig quality. No sign-accuracy claim is made without review by fluent ASL signers.</p>
    <p>Backend: <code id="api-base"></code> · data: How2Sign (CC BY-NC 4.0) or synthetic test motion · renderer: Three.js + @pixiv/three-vrm</p>
  </footer>
`;

const config = createConfig();
const canvas = must<HTMLCanvasElement>('#avatar');
const stickCanvas = must<HTMLCanvasElement>('#stick');
const sequenceSelect = must<HTMLSelectElement>('#sequence');
const playButton = must<HTMLButtonElement>('#play');
const restartButton = must<HTMLButtonElement>('#restart');
const scrub = must<HTMLInputElement>('#scrub');
const timeLabel = must<HTMLElement>('#time');
const speed = must<HTMLSelectElement>('#speed');
const loop = must<HTMLInputElement>('#loop');
const recordButton = must<HTMLButtonElement>('#record');
const avatarFile = must<HTMLInputElement>('#avatar-file');
const skin = must<HTMLInputElement>('#skin');
const background = must<HTMLInputElement>('#background');
const keypoints = must<HTMLInputElement>('#keypoints');
const mirror = must<HTMLInputElement>('#mirror');
const statusLabel = must<HTMLElement>('#status');
const banner = must<HTMLElement>('#banner');
const badges = must<HTMLElement>('#badges');
const clipTitle = must<HTMLElement>('#clip-title');
const sentenceText = must<HTMLElement>('#sentence-text');
const sentenceIndex = must<HTMLElement>('#sentence-index');
const avatarHint = must<HTMLElement>('#avatar-hint');
const faceWarning = must<HTMLElement>('#face-warning');
const faceChannels = must<HTMLElement>('#face-channels');
const debugStats = must<HTMLElement>('#debug-stats');
const debugToggle = must<HTMLButtonElement>('#debug-toggle');
const debugBody = must<HTMLElement>('#debug-body');
const debugGrid = must<HTMLElement>('#debug-grid');

must<HTMLElement>('#api-base').textContent = API_BASE;

const viewer = new AvatarViewer({
  canvas,
  stickCanvas,
  config,
  onStats: (stats) => {
    badges.textContent = `${stats.backends} · ${stats.fps.toFixed(0)} FPS · retarget ${stats.retargetMs.toFixed(2)} ms`;
    if (!scrub.matches(':active')) {
      scrub.value = String(
        stats.frameCount > 0 ? Math.round((stats.frame / Math.max(1, stats.frameCount - 1)) * 1000) : 0,
      );
    }
    const seconds = viewer.currentTime;
    timeLabel.textContent = `${seconds.toFixed(2)} / ${viewer.duration.toFixed(2)} s`;
    sentenceText.textContent = stats.sentence || '—';
    sentenceIndex.textContent = `${stats.frame + 1} / ${stats.frameCount}`;
    debugStats.textContent = JSON.stringify(viewer.debugInfo(), null, 2);
  },
});

let manifest: DemoManifest | null = null;
let currentClip: PoseSequenceData | null = null;
let recording = false;

async function bootstrap(): Promise<void> {
  try {
    manifest = await fetchManifest();
    sequenceSelect.innerHTML = '';
    for (const entry of manifest.sequences) {
      const option = document.createElement('option');
      option.value = entry.id;
      const suffix = entry.synthetic ? ' (synthetic test)' : ' (ground truth)';
      option.textContent = `${entry.title || entry.id}${suffix}`;
      sequenceSelect.append(option);
    }
    const preferred =
      manifest.realDataAvailable && manifest.sequences.find((s) => !s.synthetic)
        ? manifest.sequences.find((s) => !s.synthetic)!.id
        : manifest.sessionSequenceId ?? manifest.defaultSequenceId;
    if (preferred) sequenceSelect.value = preferred;
    await loadSequence(sequenceSelect.value);
  } catch (error) {
    sequenceSelect.innerHTML = '<option>Backend unavailable</option>';
    sequenceTextFallback(error);
  }

  const vrm = await tryLoadDefaultAvatar('/avatars/default.vrm', config);
  if (vrm) {
    viewer.setAvatar(vrm);
    updateAvatarBadges();
    viewer.play();
  }
}

async function loadSequence(id: string): Promise<void> {
  if (!id) return;
  statusLabel.textContent = `Loading ${id}…`;
  try {
    currentClip = await viewer.loadSequence(id);
    clipTitle.textContent = currentClip.title || id;
    banner.hidden = !currentClip.synthetic;
    if (currentClip.synthetic) {
      banner.textContent =
        'SYNTHETIC TEST MOTION — this is not American Sign Language. It exists to exercise the renderer without model output.';
    }
    playButton.disabled = false;
    restartButton.disabled = false;
    scrub.disabled = false;
    playButton.textContent = 'Play';
    viewer.seekFrame(0);
    viewer.play();
    statusLabel.textContent = `${currentClip.frameCount} frames @ ${currentClip.fps} fps`;
    if (currentClip.source !== 'real') {
      statusLabel.textContent += ' · no real clip found in this checkout';
    }
  } catch (error) {
    statusLabel.textContent = String(error);
  }
}

function updateAvatarBadges(): void {
  const avatar = viewer.currentAvatar;
  if (!avatar) return;
  const vrm = viewer.capabilities;
  if (vrm) {
    avatarHint.textContent = `VRM ${(avatar as { version?: string }).version ?? ''} · ${vrm.arkit.length} ARKit morphs · ${vrm.customBrows.length} custom brows`;
    if (!vrm.hasBrows) {
      faceWarning.hidden = false;
      faceWarning.textContent =
        'This avatar has no eyebrow controls. ASL yes/no vs wh-question brow marking cannot be shown faithfully.';
    } else {
      faceWarning.hidden = true;
    }
    statusLabel.textContent = vrm.summary;
  } else {
    avatarHint.textContent = 'procedural mannequin (drop a VRM to replace)';
    faceWarning.hidden = true;
  }
}

function sequenceTextFallback(error: unknown): void {
  banner.hidden = false;
  banner.innerHTML =
    `Could not reach the pose backend. Start it with <code>python -m uvicorn backend.main:app --reload</code> ` +
    `then run <code>python scripts/prepare_demo_poses.py --force</code>. (${String(error)})`;
  statusLabel.textContent = 'backend offline';
}

// ---------------------------------------------------------------- controls
sequenceSelect.addEventListener('change', () => void loadSequence(sequenceSelect.value));
playButton.addEventListener('click', () => {
  viewer.toggle();
  playButton.textContent = viewer.playing ? 'Pause' : 'Play';
});
restartButton.addEventListener('click', () => {
  viewer.seekFrame(0);
  viewer.play();
  playButton.textContent = 'Pause';
});
scrub.addEventListener('input', () => {
  const frame = Math.round((Number(scrub.value) / 1000) * Math.max(0, viewer.frameCount - 1));
  viewer.seekFrame(frame);
});
speed.addEventListener('change', () => viewer.setSpeed(Number(speed.value)));
loop.addEventListener('change', () => viewer.setLoop(loop.checked));
keypoints.addEventListener('change', () => {
  config.render.showKeypoints = keypoints.checked;
  viewer.setConfig(config);
});
mirror.addEventListener('change', () => {
  config.mirror = mirror.checked;
  viewer.setConfig(config);
  viewer.refilter();
});
skin.addEventListener('input', () => {
  config.render.skinTone = skin.value;
  const ok = viewer.setSkinTone(skin.value);
  statusLabel.textContent = ok ? 'skin tone applied' : 'this avatar exposes no tintable skin material';
});
background.addEventListener('input', () => {
  config.render.background = background.value;
  viewer.setConfig(config);
});
avatarFile.addEventListener('change', async () => {
  const file = avatarFile.files?.[0];
  if (!file) return;
  statusLabel.textContent = `Loading ${file.name}…`;
  try {
    await viewer.loadAvatarFile(file);
    updateAvatarBadges();
    statusLabel.textContent = `Loaded ${file.name}`;
    viewer.play();
  } catch (error) {
    statusLabel.textContent = `VRM load failed: ${String(error)}`;
  }
});
recordButton.addEventListener('click', async () => {
  if (!recording) {
    if (!viewer.startRecording()) {
      statusLabel.textContent = 'MediaRecorder is unavailable in this browser';
      return;
    }
    recording = true;
    recordButton.textContent = 'Stop recording';
    viewer.play();
  } else {
    const blob = await viewer.stopRecording();
    recording = false;
    recordButton.textContent = 'Record WebM';
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'contextsign-avatar.webm';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    statusLabel.textContent = `Saved ${(blob.size / 1024).toFixed(0)} KB WebM`;
  }
});
debugToggle.addEventListener('click', () => {
  debugBody.hidden = !debugBody.hidden;
  debugToggle.textContent = debugBody.hidden ? 'Debug panel ▾' : 'Debug panel ▴';
});

// ---------------------------------------------------------------- debug UI
function slider(
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  apply: (value: number) => void,
): void {
  const row = document.createElement('label');
  row.className = 'slider';
  const text = document.createElement('span');
  text.textContent = label;
  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  const out = document.createElement('code');
  out.textContent = value.toFixed(2);
  input.addEventListener('input', () => {
    const v = Number(input.value);
    out.textContent = v.toFixed(2);
    apply(v);
    viewer.setConfig(config);
  });
  row.append(text, input, out);
  debugGrid.append(row);
}

slider('Body smoothing cutoff', 0.2, 4, 0.1, config.filter.body.minCutoff, (v) => {
  config.filter.body.minCutoff = v;
  viewer.refilter();
});
slider('Hand smoothing cutoff', 0.5, 10, 0.1, config.filter.hands.minCutoff, (v) => {
  config.filter.hands.minCutoff = v;
  viewer.refilter();
});
slider('Depth hysteresis', 0, 1, 0.01, config.depth.depthFlipHysteresis, (v) => {
  config.depth.depthFlipHysteresis = v;
});
slider('Upper-arm forward prior', 0, 1, 0.01, config.depth.upperArmForward, (v) => {
  config.depth.upperArmForward = v;
});
slider('Head gain', 0, 1, 0.01, config.head.gain, (v) => {
  config.head.gain = v;
});
slider('Brow gain', 0, 2, 0.05, config.face.gain.browRaise, (v) => {
  config.face.gain.browRaise = v;
});
slider('Jaw gain', 0, 2, 0.05, config.face.gain.jawOpen, (v) => {
  config.face.gain.jawOpen = v;
});
slider('Camera distance', 2, 8, 0.1, config.camera.distance, (v) => {
  config.camera.distance = v;
});
slider('Camera target Y', -1.5, 0.5, 0.05, config.camera.targetY, (v) => {
  config.camera.targetY = v;
});

// Keep the face-channel readout alive without re-rendering the whole page.
setInterval(() => {
  const debug = viewer.debugInfo();
  faceChannels.textContent =
    `eye ${Number(debug.handConfidence).toFixed(2)} hand confidence · ` +
    `flips ${debug.flips} · head residual ${Number(debug.headResidual).toFixed(3)}`;
}, 500);

function must<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element ${selector}`);
  return element;
}

void bootstrap();

export type { AvatarConfig };
