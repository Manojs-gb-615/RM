import { createConfig, createDefaultSkeleton, filterSequence, forwardKinematics, Retargeter } from '../src/avatar';
import { NUM_POINTS } from '../src/avatar/layout';
import { readNpy } from './npy';

const raw = readNpy('../demo_data/synthetic_session.npy').data;
const cfg = createConfig();
const filtered = filterSequence(raw, 24, cfg);
const sk = createDefaultSkeleton();
const r = new Retargeter(sk, cfg);
const tips: Array<[string, number]> = [['ThumbDistal', 4], ['IndexDistal', 8], ['MiddleDistal', 12], ['RingDistal', 16], ['LittleDistal', 20]];
let worst = { e: 0, t: 0, side: 'left', bone: '', obs: [0, 0] as number[], fk: [0, 0] as number[] };
for (let t = 0; t < filtered.frames.length / (NUM_POINTS * 2); t++) {
  const frame = filtered.frames.slice(t * NUM_POINTS * 2, (t + 1) * NUM_POINTS * 2);
  const pose = r.solve(frame, 1 / 24);
  const { positions } = forwardKinematics(sk, pose.locals);
  for (const side of ['left', 'right'] as const) {
    const start = side === 'left' ? 95 : 116;
    for (const [bone, local] of tips) {
      const p = positions.get(`${side}${bone}`)!;
      const ox = frame[(start + local) * 2], oy = frame[(start + local) * 2 + 1];
      const e = Math.hypot(p.x - ox, p.y - oy);
      if (e > worst.e) worst = { e, t, side, bone, obs: [ox, oy], fk: [p.x, p.y] };
    }
  }
}
console.log(worst);
