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
  for (const bone of ['ThumbDistal', 'IndexDistal', 'MiddleDistal']) {
    const target = sol.fingerDirs.get(`${side}${bone}`)!;
    const restDir = sk.joints.get(`${side}${bone}`)!.dir;
    const mapped = restDir.clone().applyQuaternion(sol.handQuat);
    console.log(' ', bone, 'target', target.toArray().map((v) => v.toFixed(3)).join(','),
      'mappedRest', mapped.toArray().map((v) => v.toFixed(3)).join(','));
  }
}
