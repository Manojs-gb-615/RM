import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, type VRM, type VRMHumanBoneName } from '@pixiv/three-vrm';
import { Color, Mesh, Object3D, Vector3, type Material } from 'three';
import type { AvatarConfig } from './config';
import { FaceRig, type FaceCapabilities } from './faceRig';
import type { PoseFrame } from './pose';
import { allFingerBones, skeletonFromNodes, type Skeleton } from './skeleton';

const CORE_BONES = [
  'hips',
  'spine',
  'chest',
  'neck',
  'head',
  'leftShoulder',
  'leftUpperArm',
  'leftLowerArm',
  'leftHand',
  'rightShoulder',
  'rightUpperArm',
  'rightLowerArm',
  'rightHand',
];

export interface VrmLoadResult {
  avatar: VrmAvatar;
  version: string;
}

/**
 * VRM 1.0 / 0.x avatar wrapper.
 *
 * Animates three-vrm's *normalized* humanoid bones only; the library copies
 * them onto the raw bones during `vrm.update`. VRM0 models are rotated once at
 * load with the library's own helper, so no per-frame facing hacks are needed.
 */
export class VrmAvatar {
  readonly scene: Object3D;
  readonly skeleton: Skeleton;
  readonly faceRig: FaceRig;
  readonly capabilities: FaceCapabilities;
  readonly unitScale: number;
  readonly neckWorld = new Vector3();
  readonly version: string;

  private readonly nodes = new Map<string, Object3D>();
  private readonly meshes: Mesh[] = [];

  constructor(
    private readonly vrm: VRM,
    config: AvatarConfig,
  ) {
    this.scene = vrm.scene;
    this.version = String(vrm.meta?.metaVersion ?? 'unknown');

    const names = new Set([...CORE_BONES, ...allFingerBones()]);
    for (const name of names) {
      const node = vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName);
      if (node) {
        node.updateWorldMatrix(true, false);
        this.nodes.set(name, node);
      }
    }
    for (const required of CORE_BONES) {
      if (!this.nodes.has(required)) {
        // Shoulders are optional on some minimal rigs.
        if (required.includes('Shoulder')) continue;
        throw new Error(`VRM is missing required humanoid bone "${required}"`);
      }
    }

    const left = this.nodes.get('leftUpperArm');
    const right = this.nodes.get('rightUpperArm');
    const neck = this.nodes.get('neck')!;
    if (!left || !right) throw new Error('VRM is missing upper-arm bones');
    this.unitScale = left.getWorldPosition(new Vector3()).distanceTo(right.getWorldPosition(new Vector3()));
    neck.getWorldPosition(this.neckWorld);

    this.skeleton = skeletonFromNodes(this.nodes, CORE_BONES.filter((n) => this.nodes.has(n)));
    this.faceRig = new FaceRig(vrm, config);
    this.capabilities = this.faceRig.capabilities;

    this.scene.traverse((object) => {
      const mesh = object as Mesh;
      if (mesh.isMesh) {
        mesh.frustumCulled = false;
        this.meshes.push(mesh);
      }
    });
    console.info(
      `[VrmAvatar] VRM ${this.version} loaded; shoulder width ${this.unitScale.toFixed(3)} m; ` +
        `bones ${this.nodes.size}/${names.size}`,
    );
  }

  static async load(url: string, config: AvatarConfig): Promise<VrmAvatar> {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(url);
    const vrm = gltf.userData.vrm as VRM | undefined;
    if (!vrm) throw new Error('File loaded but it is not a VRM');

    // three-vrm normalises VRM0 orientation for us; never rotate the animation.
    if (vrm.meta?.metaVersion === '0') VRMUtils.rotateVRM0(vrm);
    VRMUtils.combineSkeletons(vrm.scene);

    return new VrmAvatar(vrm, config);
  }

  apply(pose: PoseFrame): void {
    for (const [name, local] of pose.locals) {
      this.nodes.get(name)?.quaternion.copy(local);
    }
    this.faceRig.apply(pose.face);
  }

  update(dt: number): void {
    this.vrm.update(dt);
  }

  /** +1 when the avatar's anatomical left is at world +X (faces +Z). */
  facingSign(): number {
    const left = this.nodes.get('leftUpperArm');
    const right = this.nodes.get('rightUpperArm');
    if (!left || !right) return 1;
    const l = left.getWorldPosition(new Vector3());
    const r = right.getWorldPosition(new Vector3());
    return Math.sign(l.x - r.x) || 1;
  }

  /** Approximate skin tinting: only materials that look like skin are touched. */
  setSkinTone(color: string): boolean {
    const tint = new Color(color);
    let matched = false;
    for (const mesh of this.meshes) {
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) {
        const m = material as Material & { color?: Color; name?: string };
        if (!m.color) continue;
        const label = `${m.name ?? ''} ${mesh.name}`.toLowerCase();
        if (/skin|face|body|arm|hand|head|neck|leg/.test(label)) {
          m.color.copy(tint);
          matched = true;
        }
      }
    }
    return matched;
  }

  dispose(): void {
    VRMUtils.deepDispose(this.vrm.scene);
  }
}

export async function tryLoadDefaultAvatar(
  url: string,
  config: AvatarConfig,
): Promise<VrmAvatar | null> {
  try {
    const head = await fetch(url, { method: 'HEAD' });
    if (!head.ok) return null;
    return await VrmAvatar.load(url, config);
  } catch {
    return null;
  }
}
