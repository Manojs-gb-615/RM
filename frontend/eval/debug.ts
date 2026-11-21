import { Vector3 } from 'three';
import { createConfig, createDefaultSkeleton, forwardKinematics, Retargeter } from '../src/avatar';
import { NUM_POINTS } from '../src/avatar/layout';
import { readNpy } from './npy';

const a = readNpy('../demo_data/synthetic_arm_raise.npy').data;
const frame = a.slice(0, NUM_POINTS * 2);
const sk = createDefaultSkeleton();
const r = new Retargeter(sk, createConfig());
const pose = r.solve(frame, 1 / 24);
const { positions } = forwardKinematics(sk, pose.locals);
for (const side of ['left', 'right'] as const) {
  const start = side === 'left' ? 95 : 116;
  console.log(side, 'wrist', positions.get(`${side}Hand`)!.toArray().map((v) => v.toFixed(3)).join(','),
    'obs', frame[start * 2].toFixed(3), frame[start * 2 + 1].toFixed(3));
  for (const [bone, local] of [
    ['ThumbDistal', 4],
    ['IndexDistal', 8],
    ['MiddleDistal', 12],
  ] as const) {
    const p = positions.get(`${side}${bone}`)!;
    console.log(' ', bone, p.toArray().map((v) => v.toFixed(3)).join(','),
      'err', Math.hypot(p.x - frame[(start + local) * 2], p.y - frame[(start + local) * 2 + 1]).toFixed(3));
  }
}
void Vector3;
