/**
 * Minimal .npy reader (v1/v2/v3 headers, little-endian float32/float64).
 * Demo poses are written by scripts/prepare_demo_poses.py with numpy.save.
 */
import { readFileSync } from 'node:fs';

export interface NpyArray {
  data: Float32Array;
  shape: number[];
  dtype: string;
}

export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.slice(1, 6));
  if (magic !== 'NUMPY') throw new Error('Not a .npy file');
  const major = bytes[6];
  let offset = 10;
  let headerLength = 0;
  if (major === 1) {
    headerLength = bytes[8] | (bytes[9] << 8);
    offset = 10;
  } else {
    headerLength =
      bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    offset = 12;
  }
  const header = new TextDecoder().decode(bytes.slice(offset, offset + headerLength));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1] ?? '<f4';
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? '';
  const shape = shapeText
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const payload = buffer.slice(offset + headerLength);
  const count = shape.reduce((a, b) => a * b, 1);
  if (descr === '<f4' || descr === '|f4') {
    return { data: new Float32Array(payload.slice(0, count * 4)), shape, dtype: descr };
  }
  if (descr === '<f8' || descr === '|f8') {
    const f64 = new Float64Array(payload.slice(0, count * 8));
    return { data: Float32Array.from(f64), shape, dtype: descr };
  }
  throw new Error(`Unsupported npy dtype ${descr}`);
}

export function readNpy(path: string): NpyArray {
  const buffer = readFileSync(path);
  return parseNpy(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}
