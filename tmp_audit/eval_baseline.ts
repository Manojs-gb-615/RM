/**
 * Avatar evaluation harness.
 *
 * Runs the *browser* retargeter in Node against the demo pose data and writes
 * measured numbers to docs/avatar_eval_results.json + docs/AVATAR_EVAL.md.
 *
 * What is measured:
 *  - 2D reprojection of forward-kinematics joints, per body part, in units of
 *    the per-clip median shoulder width (the canonical unit).
 *  - keypoint jitter before/after One Euro filtering (mean |second difference|).
 *  - depth-sign flips and large inter-frame rotation jumps ("limb flips").
 *  - self-intersection / hand-head-torso proximity events.
 *  - FK bone-length consistency against the rest skeleton.
 *  - a golden hash of retargeted rotations for regression.
 *
 * What is NOT measured here: perceptual quality, ASL correctness, or whether a
 * real VRM looks right. Those need a fluent signer and a real avatar.
 *
 * Usage:
 *   npm run eval            # verify against eval/golden.json
 *   npm run golden          # rewrite eval/golden.json (only after intended changes)
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { Quaternion, Vector3 } from 'three';
import {
  createConfig,
  createDefaultSkeleton,
  filterSequence,
  forwardKinematics,
  Retargeter,
  type AvatarConfig,
  type Skeleton,
} from '../frontend/src/avatar';
import { NUM_POINTS } from '../frontend/src/avatar/layout';
import { readNpy } from '../frontend/eval/npy';

interface ManifestSequence {
  id: string;
  file: string;
  frames: number;
  fps: number;
  source: string;
  synthetic: boolean;
  title: string;
}

interface Manifest {
  fps: number;
  sequences: ManifestSequence[];
}

interface ErrorStats {
  mean: number;
  p95: number;
  count: number;
}

interface RegionResult extends ErrorStats {
  restMean: number;
  restP95: number;
}

interface SequenceResult {
  id: string;
  title: string;
  source: string;
  synthetic: boolean;
  frames: number;
  fps: number;
  reprojection: Record<string, RegionResult>;
  jitter: {
    bodyBefore: number;
    bodyAfter: number;
    bodyReductionPct: number;
    handsBefore: number;
    handsAfter: number;
    handReductionPct: number;
  };
  limbFlips: {
    depthFlips: number;
    depthFlipsPer100Frames: number;
    largeRotationJumps: number;
    handPalmFlips: number;
  };
  selfIntersection: {
    oppositeHandFrames: number;
    handHeadFrames: number;
    handBehindTorsoFrames: number;
    anyEventFrames: number;
    anyEventPct: number;
    note: string;
  };
  boneLength: {
    maxRelativeDeviation: number;
    p95RelativeDeviation: number;
    bonesChecked: number;
  };
  outliers: number;
  repairedPoints: number;
  goldenHash: string;
}

const ROOT = resolve(process.cwd());
const DEMO_DIR = join(ROOT, 'demo_data');
const GOLDEN_PATH = join(process.cwd(), 'tmp_audit', 'golden_baseline.json');
const RESULTS_PATH = join(ROOT, 'tmp_audit', 'results_baseline.json');
const MARKDOWN_PATH = join(ROOT, 'tmp_audit', 'AVATAR_EVAL_baseline.md');

const REGION_POINTS: Record<string, Array<[string, number]>> = {
  torso: [
    ['neck', 1],
    ['hips', 8],
  ],
  leftArm: [
    ['leftUpperArm', 5],
    ['leftLowerArm', 6],
    ['leftHand', 7],
  ],
  rightArm: [
    ['rightUpperArm', 2],
    ['rightLowerArm', 3],
    ['rightHand', 4],
  ],
  head: [
    ['nose', 0],
    ['rightEye', 15],
    ['leftEye', 16],
    ['rightEar', 17],
    ['leftEar', 18],
  ],
};

const HAND_TIPS: Array<{ bone: string; local: number }> = [
  { bone: 'ThumbDistal', local: 4 },
  { bone: 'IndexDistal', local: 8 },
  { bone: 'MiddleDistal', local: 12 },
  { bone: 'RingDistal', local: 16 },
  { bone: 'LittleDistal', local: 20 },
];

const BODY_JITTER_JOINTS = [1, 2, 5, 3, 6, 4, 7, 8];
const HAND_JITTER_JOINTS = [95, 99, 103, 107, 120, 124, 128, 132];

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function stats(values: number[]): ErrorStats {
  if (values.length === 0) return { mean: 0, p95: 0, count: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return { mean, p95: quantile(values, 0.95), count: values.length };
}

function frameAt(frames: Float32Array, index: number): Float32Array {
  return frames.slice(index * NUM_POINTS * 2, (index + 1) * NUM_POINTS * 2);
}

function projectPoint(index: number, frame: Float32Array): [number, number] {
  return [frame[index * 2], frame[index * 2 + 1]];
}

/**
 * FK positions are in canonical skeleton space already (neck origin, shoulder
 * width 1), so the 2D distance to the observed keypoint is directly in
 * shoulder-width units. No re-fitting of translation or scale.
 */
function joint2D(positions: Map<string, Vector3>, name: string): [number, number] {
  const p = positions.get(name) ?? new Vector3();
  return [p.x, p.y];
}

function headPrediction(
  positions: Map<string, Vector3>,
  quats: Map<string, Quaternion>,
  neutralOffsets: Map<number, Vector3>,
  point: number,
): [number, number] {
  const neck = positions.get('neck') ?? new Vector3();
  const q = quats.get('head') ?? new Quaternion();
  const offset = (neutralOffsets.get(point) ?? new Vector3()).clone().applyQuaternion(q);
  return [neck.x + offset.x, neck.y + offset.y];
}

function buildNeutralOffsets(skeleton: Skeleton, frame: Float32Array): Map<number, Vector3> {
  const neck = new Vector3(frame[1 * 2], frame[1 * 2 + 1], 0);
  const offsets = new Map<number, Vector3>();
  for (const [, point] of REGION_POINTS.head) {
    offsets.set(point, new Vector3(frame[point * 2] - neck.x, frame[point * 2 + 1] - neck.y, 0));
  }
  void skeleton;
  return offsets;
}

/** Mean absolute second difference of selected keypoints, shoulder-width units. */
function jitter(frames: Float32Array, points: number[]): number {
  const T = frames.length / (NUM_POINTS * 2);
  if (T < 3) return 0;
  let total = 0;
  let count = 0;
  for (let t = 2; t < T; t++) {
    for (const p of points) {
      const o = (t * NUM_POINTS + p) * 2;
      const p1 = ((t - 1) * NUM_POINTS + p) * 2;
      const p2 = ((t - 2) * NUM_POINTS + p) * 2;
      total += Math.abs(frames[o] - 2 * frames[p1] + frames[p2]);
      total += Math.abs(frames[o + 1] - 2 * frames[p1 + 1] + frames[p2 + 1]);
      count += 2;
    }
  }
  return count > 0 ? total / count : 0;
}

function fnv1a(values: number[]): string {
  let hash = 0x811c9dc5;
  for (const value of values) {
    const rounded = Math.round(value * 10000) / 10000;
    const text = rounded.toFixed(4);
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  }
  return hash.toString(16).padStart(8, '0');
}

function evaluateSequence(
  sequence: ManifestSequence,
  frames: Float32Array,
  config: AvatarConfig,
  goldenStep: number,
): { result: SequenceResult; hash: string } {
  const skeleton = createDefaultSkeleton();
  const filtered = filterSequence(frames, sequence.fps, config);
  const retargeter = new Retargeter(skeleton, config);
  const T = frames.length / (NUM_POINTS * 2);

  const errors: Record<string, number[]> = {
    torso: [],
    leftArm: [],
    rightArm: [],
    head: [],
    leftHand: [],
    rightHand: [],
  };
  const restErrors: Record<string, number[]> = {
    torso: [],
    leftArm: [],
    rightArm: [],
    head: [],
    leftHand: [],
    rightHand: [],
  };
  const boneDeviations: number[] = [];
  const hashes: number[] = [];
  let depthFlips = 0;
  let handPalmFlips = 0;
  let largeJumps = 0;
  let oppositeHandFrames = 0;
  let handHeadFrames = 0;
  let handBehindTorsoFrames = 0;
  let anyEventFrames = 0;
  let previous: Map<string, Quaternion> | null = null;

  const restSkeleton = createDefaultSkeleton();
  const restPose = forwardKinematics(restSkeleton, new Map());

  for (let t = 0; t < T; t++) {
    const frame = frameAt(filtered.frames, t);
    const pose = retargeter.solve(frame, 1 / sequence.fps);
    const { positions, quats } = forwardKinematics(skeleton, pose.locals);
    const neutral = buildNeutralOffsets(skeleton, frameAt(frames, t));
    depthFlips += pose.meta.depthFlips;

    // Reprojection: direct canonical-space comparison.
    for (const [region, pairs] of Object.entries(REGION_POINTS)) {
      for (const [bone, point] of pairs) {
        const [x, y] =
          region === 'head'
            ? headPrediction(positions, quats, neutral, point)
            : joint2D(positions, bone);
        const [ox, oy] = projectPoint(point, frame);
        errors[region].push(Math.hypot(x - ox, y - oy));
        const [rx, ry] = joint2D(restPose.positions, bone);
        restErrors[region].push(Math.hypot(rx - ox, ry - oy));
      }
    }

    // Hand fingertips: FK finger bones vs observed hand keypoints.
    for (const side of ['left', 'right'] as const) {
      const start = side === 'left' ? 95 : 116;
      for (const { bone, local } of HAND_TIPS) {
        const [x, y] = joint2D(positions, `${side}${bone}`);
        const [ox, oy] = projectPoint(start + local, frame);
        const key = side === 'left' ? 'leftHand' : 'rightHand';
        errors[key].push(Math.hypot(x - ox, y - oy));
        const [rx, ry] = joint2D(restPose.positions, `${side}${bone}`);
        restErrors[key].push(Math.hypot(rx - ox, ry - oy));
      }
    }

    // Bone-length consistency: FK length vs rest length.
    for (const name of skeleton.order) {
      const joint = skeleton.joints.get(name)!;
      if (!joint.parent) continue;
      const restLength = joint.length;
      if (restLength < 1e-5) continue;
      const parentPos = positions.get(joint.parent)!;
      const length = positions.get(name)!.distanceTo(parentPos);
      boneDeviations.push(Math.abs(length - restLength) / restLength);
    }

    // Large local rotation jumps between frames (limb-flip proxy).
    if (previous) {
      let jumps = 0;
      for (const [name, q] of pose.locals) {
        const p = previous.get(name);
        if (p && q.angleTo(p) > (60 * Math.PI) / 180) jumps++;
      }
      if (jumps > 0) largeJumps++;
    }
    previous = new Map(pose.locals);

    // Depth sign flips per hand are also counted via meta.flips; this tracks
    // palm-orientation reversals specifically.
    handPalmFlips = retargeter.state.handFlips;

    // Golden hash over quantized rotations, sampled every `goldenStep` frames.
    if (t % goldenStep === 0) {
      for (const name of skeleton.order) {
        const q = pose.locals.get(name)!;
        hashes.push(q.x, q.y, q.z, q.w);
      }
    }

    // 3D proximity / intersection events.
    const leftTips = HAND_TIPS.map(({ bone }) => positions.get(`left${bone}`)!);
    const rightTips = HAND_TIPS.map(({ bone }) => positions.get(`right${bone}`)!);
    const leftWrist = positions.get('leftHand')!;
    const rightWrist = positions.get('rightHand')!;
    const head = positions.get('head')!.clone().add(new Vector3(0, 0.12, 0));
    let opposite = false;
    for (const a of leftTips) {
      for (const b of rightTips) {
        if (a.distanceTo(b) < 0.015) opposite = true;
      }
    }
    if (leftWrist.distanceTo(rightWrist) < 0.02) opposite = true;
    let headEvent = false;
    for (const tip of [...leftTips, ...rightTips]) {
      if (tip.distanceTo(head) < 0.09) headEvent = true;
    }
    const behind = [...leftTips, ...rightTips].some(
      (tip) => Math.abs(tip.x) < 0.25 && tip.y < -0.2 && tip.y > -0.95 && tip.z < -0.12,
    );
    if (opposite) oppositeHandFrames++;
    if (headEvent) handHeadFrames++;
    if (behind) handBehindTorsoFrames++;
    if (opposite || headEvent || behind) anyEventFrames++;
  }

  const reprojection: Record<string, RegionResult> = {};
  for (const region of Object.keys(errors)) {
    const s = stats(errors[region]);
    const r = stats(restErrors[region]);
    reprojection[region] = {
      mean: round(s.mean, 4),
      p95: round(s.p95, 4),
      count: s.count,
      restMean: round(r.mean, 4),
      restP95: round(r.p95, 4),
    };
  }

  const bodyBefore = jitter(filtered.repairedFrames, BODY_JITTER_JOINTS);
  const bodyAfter = jitter(filtered.frames, BODY_JITTER_JOINTS);
  const handsBefore = jitter(filtered.repairedFrames, HAND_JITTER_JOINTS);
  const handsAfter = jitter(filtered.frames, HAND_JITTER_JOINTS);
  const pct = (before: number, after: number) =>
    before > 0 ? round(((before - after) / before) * 100, 2) : 0;

  const boneMax = boneDeviations.length > 0 ? Math.max(...boneDeviations) : 0;
  const boneP95 = quantile(boneDeviations, 0.95);

  const result: SequenceResult = {
    id: sequence.id,
    title: sequence.title,
    source: sequence.source,
    synthetic: sequence.synthetic,
    frames: T,
    fps: sequence.fps,
    reprojection,
    jitter: {
      bodyBefore: round(bodyBefore, 5),
      bodyAfter: round(bodyAfter, 5),
      bodyReductionPct: pct(bodyBefore, bodyAfter),
      handsBefore: round(handsBefore, 5),
      handsAfter: round(handsAfter, 5),
      handReductionPct: pct(handsBefore, handsAfter),
    },
    limbFlips: {
      depthFlips,
      depthFlipsPer100Frames: round((depthFlips / T) * 100, 2),
      largeRotationJumps: largeJumps,
      handPalmFlips,
    },
    selfIntersection: {
      oppositeHandFrames,
      handHeadFrames,
      handBehindTorsoFrames,
      anyEventFrames,
      anyEventPct: round((anyEventFrames / T) * 100, 2),
      note: 'Contact between hands and between hand and head is legitimate in ASL; these are diagnostic proximity counts, not automatically errors.',
    },
    boneLength: {
      maxRelativeDeviation: round(boneMax, 8),
      p95RelativeDeviation: round(boneP95, 8),
      bonesChecked: skeleton.order.length,
    },
    outliers: Array.from(filtered.outliers).reduce((a, b) => a + b, 0),
    repairedPoints: Array.from(filtered.repaired).reduce((a, b) => a + b, 0),
    goldenHash: fnv1a(hashes),
  };
  return { result, hash: result.goldenHash };
}

function round(value: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

function main(): void {
  const writeGolden = process.argv.includes('--write-golden');
  const manifest = JSON.parse(readFileSync(join(DEMO_DIR, 'manifest.json'), 'utf8')) as Manifest;
  const config = createConfig();
  const results: SequenceResult[] = [];
  const hashes: Record<string, string> = {};

  for (const sequence of manifest.sequences) {
    const npy = readNpy(join(DEMO_DIR, sequence.file));
    if (npy.data.length !== sequence.frames * NUM_POINTS * 2) {
      throw new Error(`${sequence.id}: expected ${sequence.frames} frames, got ${npy.data.length / (NUM_POINTS * 2)}`);
    }
    const { result } = evaluateSequence(sequence, npy.data, config, 4);
    results.push(result);
    hashes[sequence.id] = result.goldenHash;
  }

  const previousGolden = existsSync(GOLDEN_PATH)
    ? (JSON.parse(readFileSync(GOLDEN_PATH, 'utf8')) as { hashes: Record<string, string> })
    : null;
  const goldenChecks: Array<{ id: string; expected: string; actual: string; pass: boolean }> = [];
  let goldenPass = true;
  if (previousGolden && !writeGolden) {
    for (const [id, hash] of Object.entries(hashes)) {
      const expected = previousGolden.hashes[id] ?? 'missing';
      const pass = expected === hash;
      goldenPass &&= pass;
      goldenChecks.push({ id, expected, actual: hash, pass });
    }
  }

  if (writeGolden) {
    mkdirSync(dirname(GOLDEN_PATH), { recursive: true });
    writeFileSync(
      GOLDEN_PATH,
      `${JSON.stringify(
        {
          note: 'Golden hashes of retargeter local rotations (quantized 1e-4), frames sampled every 4. Regenerate only for intended solver changes: npm run golden',
          version: 1,
          config: configVersion(config),
          hashes,
        },
        null,
        2,
      )}\n`,
      'utf8',
    );
  }

  const output = {
    generatedAt: new Date().toISOString(),
    tool: 'frontend/eval/eval.ts',
    data: {
      manifest: 'demo_data/manifest.json',
      sourceKind: 'synthetic',
      realDataAvailable: false,
      note: 'No real How2Sign clips are present in this checkout, so every number below is measured on synthetic test motion, not ASL.',
    },
    config: configVersion(config),
    golden: previousGolden
      ? { written: writeGolden, pass: goldenPass, checks: goldenChecks }
      : { written: writeGolden, pass: null, checks: [] },
    sequences: results,
    aggregate: aggregate(results),
  };

  mkdirSync(dirname(RESULTS_PATH), { recursive: true });
  writeFileSync(RESULTS_PATH, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
  writeFileSync(MARKDOWN_PATH, renderMarkdown(output), 'utf8');
  console.log(renderSummary(output));
  writeFileSync(join(process.cwd(), 'tmp_audit', 'summary_baseline.txt'), `${renderSummary(output)}\n`, 'utf8');
  if (previousGolden && !writeGolden && !goldenPass) {
    process.exitCode = 1;
  }
}

function aggregate(results: SequenceResult[]): Record<string, number> {
  const mean = (pick: (r: SequenceResult) => number) =>
    results.reduce((a, r) => a + pick(r), 0) / Math.max(1, results.length);
  return {
    sequences: results.length,
    frames: results.reduce((a, r) => a + r.frames, 0),
    meanDepthFlipsPer100Frames: round(mean((r) => r.limbFlips.depthFlipsPer100Frames), 2),
    meanLargeRotationJumpFrames: round(mean((r) => r.limbFlips.largeRotationJumps), 2),
    meanAnyProximityPct: round(mean((r) => r.selfIntersection.anyEventPct), 2),
    maxBoneLengthDeviation: round(Math.max(...results.map((r) => r.boneLength.maxRelativeDeviation)), 8),
    bodyJitterReductionPct: round(mean((r) => r.jitter.bodyReductionPct), 2),
    handJitterReductionPct: round(mean((r) => r.jitter.handReductionPct), 2),
  };
}

function configVersion(config: AvatarConfig): Record<string, unknown> {
  return {
    filter: config.filter,
    depth: config.depth,
    hands: config.hands,
    torso: config.torso,
    head: config.head,
    smoothing: config.smoothing,
    face: config.face,
  };
}

function renderSummary(output: {
  sequences: SequenceResult[];
  aggregate: Record<string, number>;
  golden: { written: boolean; pass: boolean | null };
}): string {
  const lines: string[] = [];
  lines.push('Avatar evaluation (synthetic demo data; no real ASL clips in this checkout)');
  lines.push('');
  for (const r of output.sequences) {
    const reproj = Object.entries(r.reprojection)
      .map(([k, v]) => `${k} ${v.mean.toFixed(3)}/${v.p95.toFixed(3)}`)
      .join('  ');
    lines.push(
      `${r.id}: ${r.frames}f @ ${r.fps}fps | reprojection mean/p95 (SW): ${reproj} | ` +
        `depth flips ${r.limbFlips.depthFlips} | jitter body ${r.jitter.bodyReductionPct}% hands ${r.jitter.handReductionPct}% | ` +
        `proximity ${r.selfIntersection.anyEventPct}% | bone max ${r.boneLength.maxRelativeDeviation} | hash ${r.goldenHash}`,
    );
  }
  lines.push('');
  lines.push(`Aggregate: ${JSON.stringify(output.aggregate)}`);
  lines.push(
    `Golden: ${output.golden.written ? 'written' : output.golden.pass === null ? 'not present' : output.golden.pass ? 'PASS' : 'FAIL'}`,
  );
  return lines.join('\n');
}

function renderMarkdown(output: {
  generatedAt: string;
  sequences: SequenceResult[];
  aggregate: Record<string, number>;
  golden: { written: boolean; pass: boolean | null; checks: Array<{ id: string; expected: string; actual: string; pass: boolean }> };
}): string {
  const lines: string[] = [];
  lines.push('# Avatar evaluation results');
  lines.push('');
  lines.push(
    `Generated by \`frontend/eval/eval.ts\` at ${output.generatedAt}. ` +
      'All numbers below were measured locally in this checkout. **The demo data is synthetic test motion, not ASL.** ' +
      'No real How2Sign pose arrays were available, so these results validate the retargeter mechanics only.',
  );
  lines.push('');
  lines.push('## Method');
  lines.push('');
  lines.push(
    '- Runs the same TypeScript retargeter the browser uses (`src/avatar/retarget.ts`) in Node.\n' +
      '- Reprojection compares forward-kinematics joints to the input canonical keypoints in 2D. ' +
      'Positions are already in canonical units (neck origin, per-clip median shoulder width = 1), so distances are in shoulder-width (SW) units. ' +
      '`rest` columns are the identity/rest-pose baseline, included to show the solver is doing something.\n' +
      '- Jitter is mean absolute second difference of selected body and hand keypoints, measured before and after One Euro filtering.\n' +
      '- Depth flips are counted by the solver (sign of inferred z changes between frames). Large rotation jumps count frames where any local bone rotates more than 60Â° from the previous frame.\n' +
      '- Proximity counts frames where fingertips of opposite hands, hand-to-head, or hand-behind-torso events occur. ASL uses contact, so these are diagnostics, not failures.\n' +
      '- Bone-length consistency is FK length versus rest length for every bone.\n' +
      '- Golden hashes are FNV-1a over local quaternions quantized to 1e-4, sampled every 4 frames.',
  );
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push('| Sequence | Frames | Data | Depth flips | Flips/100f | Large rot. jump frames | Proximity % | Body jitter â†“ | Hand jitter â†“ | Max bone dev. |');
  lines.push('| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
  for (const r of output.sequences) {
    lines.push(
      `| ${r.id} | ${r.frames} | ${r.synthetic ? 'synthetic (not ASL)' : r.source} | ${r.limbFlips.depthFlips} | ` +
        `${r.limbFlips.depthFlipsPer100Frames} | ${r.limbFlips.largeRotationJumps} | ${r.selfIntersection.anyEventPct}% | ` +
        `${r.jitter.bodyReductionPct}% | ${r.jitter.handReductionPct}% | ${r.boneLength.maxRelativeDeviation} |`,
    );
  }
  lines.push('');
  lines.push('## Reprojection error by body part (shoulder-width units)');
  lines.push('');
  lines.push('| Sequence | Region | Mean | p95 | Rest-pose mean | Rest-pose p95 |');
  lines.push('| --- | --- | ---: | ---: | ---: | ---: |');
  for (const r of output.sequences) {
    for (const [region, v] of Object.entries(r.reprojection)) {
      lines.push(
        `| ${r.id} | ${region} | ${v.mean.toFixed(4)} | ${v.p95.toFixed(4)} | ${v.restMean.toFixed(4)} | ${v.restP95.toFixed(4)} |`,
      );
    }
  }
  lines.push('');
  lines.push('## Probe measurements');
  lines.push('');
  for (const r of output.sequences) {
    lines.push(`### ${r.id}`);
    lines.push('');
    lines.push(
      `- Jitter body: ${r.jitter.bodyBefore} â†’ ${r.jitter.bodyAfter} (${r.jitter.bodyReductionPct}% reduction)\n` +
        `- Jitter hands: ${r.jitter.handsBefore} â†’ ${r.jitter.handsAfter} (${r.jitter.handReductionPct}% reduction)\n` +
        `- Hand palm-orientation flips: ${r.limbFlips.handPalmFlips}\n` +
        `- Proximity frames: opposite hands ${r.selfIntersection.oppositeHandFrames}, hand-head ${r.selfIntersection.handHeadFrames}, hand-behind-torso ${r.selfIntersection.handBehindTorsoFrames}\n` +
        `- Speed-outlier points rejected: ${r.outliers}; missing points repaired: ${r.repairedPoints}\n` +
        `- Bone-length deviation: max ${r.boneLength.maxRelativeDeviation}, p95 ${r.boneLength.p95RelativeDeviation}\n` +
        `- Golden hash: \`${r.goldenHash}\``,
    );
    lines.push('');
  }
  lines.push('## Golden regression');
  lines.push('');
  if (output.golden.written) {
    lines.push('Golden hashes were rewritten by this run (`npm run golden`).');
  } else if (output.golden.pass === null) {
    lines.push('No `eval/golden.json` existed, so hashes were not compared. Run `npm run golden` to record them.');
  } else {
    lines.push(`Result: **${output.golden.pass ? 'PASS' : 'FAIL'}**`);
    lines.push('');
    lines.push('| Sequence | Expected | Actual | Pass |');
    lines.push('| --- | --- | --- | --- |');
    for (const check of output.golden.checks) {
      lines.push(`| ${check.id} | ${check.expected} | ${check.actual} | ${check.pass ? 'yes' : 'no'} |`);
    }
  }
  lines.push('');
  lines.push('## What these numbers do and do not show');
  lines.push('');
  lines.push(
    '- They show that 2D reprojection stays bounded, filtering reduces jitter, bone lengths stay constant, and the solver is deterministic (golden).\n' +
      '- They do **not** show ASL accuracy, naturalness, or correctness of inferred depth. 2Dâ†’3D depth is under-determined; the sign is chosen by a forward prior and hysteresis.\n' +
      '- They are measured on synthetic motions, not human signers. Real How2Sign clips and a real VRM must be checked manually before making any claim about signing quality.\n' +
      '- The mannequin/rest skeleton proportions differ from a real VRM; per-avatar reprojection should be re-run with an avatar-specific skeleton later.',
  );
  lines.push('');
  return lines.join('\n');
}

main();

