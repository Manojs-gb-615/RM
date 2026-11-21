import type { VRM } from '@pixiv/three-vrm';
import type { Mesh } from 'three';
import type { AvatarConfig } from './config';
import { clamp } from './math3';
import type { FaceChannels } from './faceChannels';

const ARKIT_BROWS = [
  'browInnerUp',
  'browDownLeft',
  'browDownRight',
  'browOuterUpLeft',
  'browOuterUpRight',
];

const ARKIT_EYES = ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookDownLeft', 'eyeLookUpRight', 'eyeLookDownRight'];

const ARKIT_MOUTH = [
  'jawOpen',
  'mouthFunnel',
  'mouthPucker',
  'mouthSmileLeft',
  'mouthSmileRight',
  'mouthFrownLeft',
  'mouthFrownRight',
  'cheekPuff',
];

const PRESETS = ['aa', 'ih', 'ou', 'ee', 'oh', 'blink', 'happy', 'angry', 'sad', 'lookUp', 'lookDown', 'lookLeft', 'lookRight'];

export interface FaceCapabilities {
  arkit: string[];
  presets: string[];
  customBrows: string[];
  hasBrows: boolean;
  hasExpressions: boolean;
  /** Human-readable summary logged at load and shown in the demo. */
  summary: string;
}

interface MorphBinding {
  mesh: Mesh & { morphTargetInfluences?: number[] };
  index: number;
}

/**
 * FaceRig drives either ARKit-52 morph targets or VRM expression presets.
 *
 * Capability detection happens once at avatar load. Every unsupported channel
 * is reported; if no brow control exists the demo shows a visible warning,
 * because ASL yes/no vs wh-question marking depends on the brows.
 */
export class FaceRig {
  readonly capabilities: FaceCapabilities;
  private readonly morphs = new Map<string, MorphBinding[]>();
  private readonly config: AvatarConfig;
  private readonly spoke = new Set<string>();

  constructor(
    private readonly vrm: VRM,
    config: AvatarConfig,
  ) {
    this.config = config;
    const morphNames = new Set<string>();
    vrm.scene.traverse((object) => {
      const mesh = object as Mesh & {
        morphTargetDictionary?: Record<string, number>;
        morphTargetInfluences?: number[];
      };
      if (!mesh.isMesh || !mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;
      for (const [name, index] of Object.entries(mesh.morphTargetDictionary)) {
        morphNames.add(name);
        const list = this.morphs.get(name) ?? [];
        list.push({ mesh, index });
        this.morphs.set(name, list);
      }
    });

    const expressionMap = vrm.expressionManager?.expressionMap ?? {};
    const expressions = Object.keys(expressionMap);
    const arkit = [...ARKIT_BROWS, ...ARKIT_EYES, ...ARKIT_MOUTH].filter((name) =>
      morphNames.has(name),
    );
    const presets = PRESETS.filter(
      (name) => expressionMap[name] !== undefined || morphNames.has(name),
    );
    const customBrows = expressions.filter(
      (name) => /brow|eyebrow/i.test(name) && !ARKIT_BROWS.includes(name),
    );
    const hasBrows = ARKIT_BROWS.some((name) => morphNames.has(name)) || customBrows.length > 0;

    this.capabilities = {
      arkit,
      presets,
      customBrows,
      hasBrows,
      hasExpressions: expressions.length > 0,
      summary: describe(arkit, presets, customBrows, hasBrows),
    };
    console.info('[FaceRig]', this.capabilities.summary);
  }

  apply(channels: FaceChannels): void {
    const g = this.config.face.gain;
    const browRaise = clamp(channels.browRaise * 12 * g.browRaise, 0, 1);
    const browFurrow = clamp(channels.browFurrow * 8 * g.browFurrow, 0, 1);
    const blink = clamp((1 - channels.eyeOpen) * g.blink, 0, 1);
    const jaw = clamp(channels.jawOpen * 12 * g.jawOpen, 0, 1);
    const smile = clamp(channels.smile * 16 * g.smile, 0, 1);
    const frown = clamp(channels.frown * 16 * g.frown, 0, 1);
    const wide = clamp(Math.max(0, channels.mouthWide) * 3 * g.mouthWide, 0, 1);
    const pucker = clamp(Math.max(0, channels.mouthRound) * 1.5 * g.mouthRound, 0, 1);
    const gazeX = clamp(channels.gazeX * 3 * g.gaze, -1, 1);
    const gazeY = clamp(channels.gazeY * 3 * g.gaze, -1, 1);

    if (this.hasMorph('browInnerUp')) {
      this.setMorph('browInnerUp', browRaise);
      this.setMorph('browOuterUpLeft', browRaise * 0.8);
      this.setMorph('browOuterUpRight', browRaise * 0.8);
    } else {
      this.applyCustomBrows(browRaise);
      this.setExpression('angry', browFurrow * 0.35);
    }
    this.setMorph('browDownLeft', browFurrow);
    this.setMorph('browDownRight', browFurrow);

    if (this.hasMorph('eyeBlinkLeft')) {
      this.setMorph('eyeBlinkLeft', blink);
      this.setMorph('eyeBlinkRight', blink);
    } else {
      this.setExpression('blink', blink);
    }
    this.applyGaze(gazeX, gazeY);

    this.setMorph('jawOpen', jaw);
    this.setMorph('mouthSmileLeft', smile);
    this.setMorph('mouthSmileRight', smile);
    this.setMorph('mouthFrownLeft', frown);
    this.setMorph('mouthFrownRight', frown);
    this.setMorph('mouthPucker', pucker);
    this.setMorph('mouthFunnel', pucker * 0.4);
    this.setMorph('cheekPuff', 0);

    // VRM presets only fill channels that have no ARKit morph.
    if (!this.hasMorph('jawOpen')) {
      this.setExpression('aa', jaw * 0.85);
      this.setExpression('oh', jaw * 0.25 + pucker * 0.3);
    }
    if (!this.hasMorph('mouthPucker')) this.setExpression('ou', pucker * 0.8);
    if (!this.hasMorph('mouthSmileLeft')) this.setExpression('happy', smile * 0.8);
    if (wide > 0) this.setExpression('ih', wide * 0.4);
    this.setExpression('sad', frown * 0.5);
  }

  private applyCustomBrows(weight: number): void {
    if (this.capabilities.customBrows.length === 0) return;
    for (const name of this.capabilities.customBrows) this.setExpression(name, weight);
  }

  private applyGaze(x: number, y: number): void {
    if (this.hasMorph('eyeLookOutLeft')) {
      // Positive x means the signer looks toward their anatomical left.
      this.setMorph('eyeLookOutLeft', clamp(x, 0, 1));
      this.setMorph('eyeLookInLeft', clamp(-x, 0, 1));
      this.setMorph('eyeLookInRight', clamp(x, 0, 1));
      this.setMorph('eyeLookOutRight', clamp(-x, 0, 1));
      this.setMorph('eyeLookUpLeft', clamp(y, 0, 1));
      this.setMorph('eyeLookDownLeft', clamp(-y, 0, 1));
      this.setMorph('eyeLookUpRight', clamp(y, 0, 1));
      this.setMorph('eyeLookDownRight', clamp(-y, 0, 1));
    } else {
      this.setExpression('lookLeft', clamp(x, 0, 1));
      this.setExpression('lookRight', clamp(-x, 0, 1));
      this.setExpression('lookUp', clamp(y, 0, 1));
      this.setExpression('lookDown', clamp(-y, 0, 1));
    }
  }

  private hasMorph(name: string): boolean {
    return this.morphs.has(name);
  }

  private setMorph(name: string, weight: number): void {
    const list = this.morphs.get(name);
    if (!list) {
      this.warnOnce(name);
      return;
    }
    const w = clamp(weight, 0, 1);
    for (const binding of list) {
      if (binding.mesh.morphTargetInfluences) {
        binding.mesh.morphTargetInfluences[binding.index] = w;
      }
    }
  }

  private setExpression(name: string, weight: number): void {
    const manager = this.vrm.expressionManager;
    if (!manager) return;
    if (manager.getExpression(name) === null) return;
    manager.setValue(name, clamp(weight, 0, 1));
  }

  private warnOnce(name: string): void {
    if (this.spoke.has(name) || this.spoke.size > 6) return;
    this.spoke.add(name);
    console.info(`[FaceRig] Avatar has no "${name}" channel; skipping.`);
  }
}

function describe(
  arkit: string[],
  presets: string[],
  customBrows: string[],
  hasBrows: boolean,
): string {
  const parts = [
    `ARKit morphs: ${arkit.length}`,
    `VRM presets: ${presets.join(', ') || 'none'}`,
    `custom brows: ${customBrows.join(', ') || 'none'}`,
  ];
  parts.push(
    hasBrows
      ? 'Brow channels detected — question marking is available.'
      : 'NO BROW CHANNELS — yes/no vs wh-question marking will not be visible.',
  );
  return parts.join(' | ');
}
