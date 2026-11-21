import { Vector3 } from 'three';
import { createConfig, createDefaultSkeleton } from '../src/avatar';
import { NUM_POINTS } from '../src/avatar/layout';
import { solveHand, createHandState } from '../src/avatar/handSolver';
import { readNpy } from './npy';

const a = readNpy('../demo_data/synthetic_arm_raise.npy').data;
const frame = a.slice(0, NUM_POINTS * 2);
const sk = createDefaultSkeleton();
for (const side of ['left', 'right'] as const) {
  const start = side === 'left' ? 95 : 116;
  const sol = solveHand(frame, start, side, side, new Vector3(0, -1, 0), sk, createConfig(), createHandState(), 1);
  console.log(side, 'handQuat', sol.handQuat.toArray().map((v) => v.toFixed(3)).join(','));
  console.log('keys', [...sol.fingerDirs.keys()]);
  for (const [bone, local] of [
    ['ThumbDistal', 4],
    ['IndexDistal', 8],
    ['MiddleDistal', 12],
  ] as const) {
    const target = sol.fingerDirs.get(`${side}${bone}`)!;
    const restDir = sk.joints.get(`${side}${bone}`)!.dir;
    console.log(' ', bone, 'target', target.toArray().map((v) => v.toFixed(3)).join(','),
      'rest', restDir.toArray().map((v) => v.toFixed(3)).join(','),
      'obs delta', (frame[(start + local) * 2] - frame[start * 2]).toFixed(3), (frame[(start + local) * 2 + 1] - frame[start * 2 + 1]).toFixed(3));
  }
}
