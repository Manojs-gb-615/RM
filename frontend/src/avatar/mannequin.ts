import {
  BoxGeometry,
  Color,
  CylinderGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  SphereGeometry,
  Vector3,
} from 'three';
import type { PoseFrame } from './pose';
import { createDefaultSkeleton, type Skeleton } from './skeleton';

/**
 * Procedural fallback avatar: a simple humanoid with the same humanoid
 * skeleton as VRM and a minimal face (brows, eyes, mouth, pupils). It exists so
 * the demo, tests and evaluation work without any downloaded avatar asset.
 */
export class MannequinAvatar {
  readonly root = new Group();
  readonly skeleton: Skeleton = createDefaultSkeleton();
  readonly unitScale = 0.42; // metres per shoulder width
  readonly neckWorld = new Vector3(0, 1.35, 0);
  readonly faceSupported = true;
  readonly capabilitiesSummary = 'Procedural mannequin face (brows, lids, mouth, gaze).';

  private readonly nodes = new Map<string, Object3D>();
  private readonly skinMaterial = new MeshStandardMaterial({
    color: new Color('#c98d6b'),
    roughness: 0.85,
    metalness: 0,
  });
  private readonly darkMaterial = new MeshStandardMaterial({ color: '#1b1f27', roughness: 0.6 });
  private readonly eyeMaterial = new MeshStandardMaterial({ color: '#f4f6f8', roughness: 0.4 });
  private readonly pupils: Mesh[] = [];
  private readonly eyes: Mesh[] = [];
  private readonly brows: Mesh[] = [];
  private mouth?: Mesh;
  private mouthBaseScale = new Vector3(1, 1, 1);

  constructor() {
    this.buildSkeletonNodes();
    this.buildBody();
    this.buildFace();
    this.root.scale.setScalar(this.unitScale);
    this.root.position.copy(this.neckWorld);
  }

  get scene(): Object3D {
    return this.root;
  }

  private buildSkeletonNodes(): void {
    for (const name of this.skeleton.order) {
      const joint = this.skeleton.joints.get(name)!;
      const node = new Object3D();
      node.name = name;
      if (joint.parent) {
        const parentNode = this.nodes.get(joint.parent)!;
        node.position.copy(joint.pos).sub(this.skeleton.joints.get(joint.parent)!.pos);
        parentNode.add(node);
      } else {
        node.position.copy(joint.pos);
        this.root.add(node);
      }
      this.nodes.set(name, node);
    }
  }

  private buildBody(): void {
    const cylinder = new CylinderGeometry(1, 1, 1, 10);
    cylinder.translate(0, 0.5, 0);
    const sphere = new SphereGeometry(1, 12, 10);

    for (const name of this.skeleton.order) {
      const joint = this.skeleton.joints.get(name)!;
      if (!joint.parent) continue;
      const parent = this.skeleton.joints.get(joint.parent)!;
      const offset = joint.pos.clone().sub(parent.pos);
      const length = offset.length();
      if (length < 1e-4) continue;
      const isFinger = /Thumb|Index|Middle|Ring|Little/.test(name);
      const isArm = /UpperArm|LowerArm/.test(name);
      const radius = isFinger ? 0.018 : isArm ? 0.045 : /Hand/.test(name) ? 0.03 : 0.06;
      const mesh = new Mesh(cylinder, this.skinMaterial);
      mesh.scale.set(radius, length, radius);
      mesh.position.copy(offset).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), offset.clone().normalize());
      this.nodes.get(joint.parent)!.add(mesh);

      if (!isFinger) {
        const jointMesh = new Mesh(sphere, this.skinMaterial);
        jointMesh.scale.setScalar(radius * 1.15);
        this.nodes.get(name)!.add(jointMesh);
      }
    }

    const head = this.nodes.get('head');
    if (head) {
      const skull = new Mesh(sphere, this.skinMaterial);
      skull.scale.set(0.19, 0.23, 0.19);
      skull.position.set(0, 0.17, 0.01);
      head.add(skull);
    }
  }

  private buildFace(): void {
    const head = this.nodes.get('head');
    if (!head) return;
    const eyeGeometry = new SphereGeometry(0.028, 12, 10);
    const pupilGeometry = new SphereGeometry(0.011, 8, 8);
    const browGeometry = new BoxGeometry(0.06, 0.012, 0.012);
    for (const sign of [-1, 1]) {
      const eye = new Mesh(eyeGeometry, this.eyeMaterial);
      eye.position.set(0.062 * sign, 0.19, 0.16);
      head.add(eye);
      this.eyes.push(eye);
      const pupil = new Mesh(pupilGeometry, this.darkMaterial);
      pupil.position.set(0, 0, 0.023);
      eye.add(pupil);
      this.pupils.push(pupil);

      const brow = new Mesh(browGeometry, this.darkMaterial);
      brow.position.set(0.062 * sign, 0.245, 0.165);
      head.add(brow);
      this.brows.push(brow);
    }
    this.mouth = new Mesh(new BoxGeometry(0.075, 0.016, 0.012), this.darkMaterial);
    this.mouth.position.set(0, 0.105, 0.165);
    this.mouthBaseScale = this.mouth.scale.clone();
    head.add(this.mouth);
  }

  apply(pose: PoseFrame): void {
    for (const [name, local] of pose.locals) {
      this.nodes.get(name)?.quaternion.copy(local);
    }
    const channels = pose.face;
    for (const brow of this.brows) {
      brow.position.y = 0.245 + channels.browRaise * 0.7;
      brow.rotation.z = 0;
    }
    for (const eye of this.eyes) {
      eye.scale.y = Math.max(0.08, channels.eyeOpen);
    }
    for (const pupil of this.pupils) {
      pupil.position.x = channels.gazeX * 0.012;
      pupil.position.y = channels.gazeY * 0.01;
    }
    if (this.mouth) {
      const jaw = Math.min(0.08, channels.jawOpen * 0.9);
      this.mouth.scale.set(
        this.mouthBaseScale.x * (1 + Math.max(0, channels.mouthWide) * 0.4 - Math.max(0, channels.mouthRound) * 0.35),
        this.mouthBaseScale.y * (1 + jaw * 8),
        this.mouthBaseScale.z,
      );
      this.mouth.position.y = 0.105 - jaw * 0.4;
    }
  }

  setSkinTone(color: string): boolean {
    this.skinMaterial.color.set(color);
    return true;
  }

  update(_dt: number): void {
    // Procedural avatar has no spring bones or expression manager.
  }

  /** World-space facing sign: +1 faces +Z, -1 faces -Z. */
  facingSign(): number {
    const left = this.skeleton.joints.get('leftUpperArm')!.pos.x;
    const right = this.skeleton.joints.get('rightUpperArm')!.pos.x;
    return Math.sign(left - right) || 1;
  }

  dispose(): void {
    this.root.traverse((object) => {
      if (object instanceof Mesh) {
        object.geometry.dispose();
      }
    });
    this.skinMaterial.dispose();
    this.darkMaterial.dispose();
    this.eyeMaterial.dispose();
  }
}
