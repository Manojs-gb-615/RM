import { createConfig, createDefaultSkeleton, filterSequence, forwardKinematics, Retargeter } from '../src/avatar';
import { NUM_POINTS } from '../src/avatar/layout';
import { readNpy } from './npy';

const raw = readNpy('../demo_data/synthetic_session.npy').data;
const cfg = createConfig();
const filtered = filterSequence(raw, 24, cfg);
const sk = createDefaultSkeleton();
const r = new Retargeter(sk, cfg);
for (let t = 0; t < 70; t++) {
  r.solve(filtered.frames.slice(t * NUM_POINTS * 2, (t + 1) * NUM_POINTS * 2), 1 / 24);
}
const t = 70;
const frame = filtered.frames.slice(t * NUM_POINTS * 2, (t + 1) * NUM_POINTS * 2);
const pose = r.solve(frame, 1 / 24);
const { positions } = forwardKinematics(sk, pose.locals);
for (const bone of ['leftHand', 'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal']) {
  const q = pose.locals.get(bone)!;
  const p = positions.get(bone)!;
  console.log(bone, 'q', q.toArray().map((v) => v.toFixed(3)).join(','), 'pos', p.toArray().map((v) => v.toFixed(3)).join(','));
}
console.log('obs tips', [16, 17, 18, 19, 20].map((i) => frame[(95 + i) * 2].toFixed(3) + ',' + frame[(95 + i) * 2 + 1].toFixed(3)).join(' | '));
