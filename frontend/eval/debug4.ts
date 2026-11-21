import { Quaternion, Vector3 } from 'three';
import { safeNormalize, quatFromBasis } from '../src/avatar/math3';
import { createDefaultSkeleton } from '../src/avatar';

const sk = createDefaultSkeleton();
for (const side of ['left', 'right'] as const) {
  const hand = sk.joints.get(`${side}Hand`)!;
  const idx = sk.joints.get(`${side}IndexProximal`)!;
  const pinky = sk.joints.get(`${side}LittleProximal`)!;
  const f = safeNormalize(idx.pos.clone().sub(hand.pos));
  const lat = safeNormalize(idx.pos.clone().sub(pinky.pos));
  const n = new Vector3().crossVectors(f, lat).normalize();
  const l = new Vector3().crossVectors(n, f).normalize();
  const q = quatFromBasis(f, l, n);
  console.log(side, 'f', f.toArray(), 'lat', lat.toArray(), 'n', n.toArray(), 'l', l.toArray());
  console.log('  rest q', q.toArray(), 'len', q.length());
}
const fDir = new Vector3(-0.697, -0.712, 0.05).normalize();
const normal = new Vector3(0, 0, -1);
const latOrtho = new Vector3().crossVectors(normal, fDir).normalize();
const qt = quatFromBasis(fDir, latOrtho, normal);
console.log('target q', qt.toArray(), 'len', qt.length(), 'latOrtho', latOrtho.toArray());
const qrest = quatFromBasis(new Vector3(-0.958, 0, 0.287), new Vector3(0.287, 0, 0.958), new Vector3(0, 1, 0));
const h = qt.clone().multiply(qrest.clone().invert());
console.log('hand q', h.toArray(), 'len', h.length());
