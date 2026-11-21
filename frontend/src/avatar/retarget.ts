import { Quaternion, Vector3 } from 'three';
import type { AvatarConfig } from './config';
import { solveArm } from './depth';
import {
  extractFaceChannels,
  NeutralCalibrator,
  NEUTRAL_FACE,
  type FaceChannels,
} from './faceChannels';
import { createHandState, solveHand, type HandState } from './handSolver';
import { fallbackHeadPose, solveHead } from './headSolver';
import { IDX } from './layout';
import { clamp, halfLifeAlpha, safeNormalize } from './math3';
import type { PoseFrame, PoseMeta } from './pose';
import {
  allFingerBones,
  FINGERS,
  fingerBoneNames,
  SIDES,
  type Side,
  type Skeleton,
} from './skeleton';

const X = new Vector3(1, 0, 0);
const Y = new Vector3(0, 1, 0);
const Z = new Vector3(0, 0, 1);

const DATA = {
  left: { shoulder: IDX.leftShoulder, elbow: IDX.leftElbow, wrist: IDX.leftWrist, hand: IDX.leftHandStart },
  right: { shoulder: IDX.rightShoulder, elbow: IDX.rightElbow, wrist: IDX.rightWrist, hand: IDX.rightHandStart },
} as const;

interface ArmState {
  elbowZ: number;
  wristZ: number;
}

export interface RetargetState {
  arms: Record<Side, ArmState>;
  hands: Record<Side, HandState>;
  flips: number;
  handFlips: number;
  elbowClamped: number;
  headResidual: number;
  handConfidence: number;
}

export interface RetargetOptions {
  /** Called once per solved frame with the calibrated face channels. */
  onFace?: (channels: FaceChannels) => void;
}

/**
 * Browser-side retargeter: canonical 2D keypoints -> local quaternions for a
 * humanoid skeleton. Pure math, no renderer dependency, so the same class runs
 * in the demo and in the Node evaluation harness.
 */
export class Retargeter {
  readonly state: RetargetState = {
    arms: { left: { elbowZ: 0, wristZ: 0 }, right: { elbowZ: 0, wristZ: 0 } },
    hands: { left: createHandState(), right: createHandState() },
    flips: 0,
    handFlips: 0,
    elbowClamped: 0,
    headResidual: 0,
    handConfidence: 1,
  };

  private previousLocals = new Map<string, Quaternion>();
  private previousFace: FaceChannels = { ...NEUTRAL_FACE };
  private calibrator: NeutralCalibrator;
  private readonly fingerSet = new Set(allFingerBones());

  constructor(
    private readonly skeleton: Skeleton,
    private config: AvatarConfig,
  ) {
    this.calibrator = new NeutralCalibrator(config.face.neutralFrames);
  }

  setConfig(config: AvatarConfig): void {
    this.config = config;
    if (config.face.neutralFrames !== this.calibrator.configuredFrames) {
      this.calibrator = new NeutralCalibrator(config.face.neutralFrames);
    }
  }

  reset(): void {
    this.previousLocals.clear();
    this.previousFace = { ...NEUTRAL_FACE };
    this.calibrator.reset();
    this.state.arms.left = { elbowZ: 0, wristZ: 0 };
    this.state.arms.right = { elbowZ: 0, wristZ: 0 };
    this.state.hands = { left: createHandState(), right: createHandState() };
    this.state.flips = 0;
    this.state.handFlips = 0;
    this.state.elbowClamped = 0;
    this.state.headResidual = 0;
    this.state.handConfidence = 1;
  }

  solve(frame: Float32Array, dt = 1 / 24): PoseFrame {
    const config = this.config;
    const skeleton = this.skeleton;
    const mirror = config.mirror;
    const xSign = mirror ? -1 : 1;
    const world = new Map<string, Quaternion>();
    const setWorld = (name: string, q: Quaternion) => world.set(name, q);

    // Rest local rotations are identity in normalized VRM space, so the world
    // rotation below is also the local delta for a root-level bone.
    let flips = 0;
    let handFlips = 0;
    let elbowClamped = 0;
    let handConfidence = 1;

    // ---- torso -----------------------------------------------------------
    const sx = (i: number) => frame[i * 2] * xSign;
    const sy = (i: number) => frame[i * 2 + 1];
    const shoulderL = new Vector3(sx(IDX.leftShoulder), sy(IDX.leftShoulder), 0);
    const shoulderR = new Vector3(sx(IDX.rightShoulder), sy(IDX.rightShoulder), 0);
    const shoulderLine = shoulderL.clone().sub(shoulderR);
    let roll = Math.atan2(shoulderLine.y, shoulderLine.x);
    if (Math.abs(roll) > Math.PI / 2) roll -= Math.sign(roll) * Math.PI;
    roll = clamp(roll * config.torso.rollGain, -config.torso.maxRoll, config.torso.maxRoll);

    const shoulderMid = shoulderL.clone().add(shoulderR).multiplyScalar(0.5);
    const yaw = clamp(
      (sx(IDX.nose) - shoulderMid.x) * config.torso.yawGain,
      -config.torso.maxYaw,
      config.torso.maxYaw,
    );
    const hip = new Vector3(sx(IDX.midHip), sy(IDX.midHip), 0);
    const neck = new Vector3(sx(IDX.neck), sy(IDX.neck), 0);
    const torsoVector = hip.clone().sub(neck).normalize();
    const pitch = clamp(
      Math.atan2(torsoVector.x, Math.max(1e-4, -torsoVector.y)) * config.torso.pitchGain,
      -config.torso.maxPitch,
      config.torso.maxPitch,
    );

    const qRoll = new Quaternion().setFromAxisAngle(Z, roll);
    const qYaw = new Quaternion().setFromAxisAngle(Y, yaw);
    const qPitch = new Quaternion().setFromAxisAngle(X, pitch);
    const torso = qYaw.multiply(qPitch).multiply(qRoll);
    const torsoAt = (weight: number) => new Quaternion().slerp(torso, weight);

    setWorld('hips', new Quaternion());
    setWorld('spine', torsoAt(0.3));
    setWorld('chest', torsoAt(0.6));
    setWorld('neck', torsoAt(0.8));
    setWorld('leftShoulder', world.get('chest')!.clone());
    setWorld('rightShoulder', world.get('chest')!.clone());

    // ---- arms ------------------------------------------------------------
    for (const side of SIDES) {
      const dataSide = mirror ? (side === 'left' ? 'right' : 'left') : side;
      const data = DATA[dataSide];
      const bones = [`${side}UpperArm`, `${side}LowerArm`, `${side}Hand`] as const;
      const upperJoint = skeleton.joints.get(bones[0]);
      const foreJoint = skeleton.joints.get(bones[1]);
      const handJoint = skeleton.joints.get(bones[2]);
      if (!upperJoint || !foreJoint || !handJoint) continue;

      const armState = this.state.arms[side];
      const arm = solveArm(
        upperJoint.pos.clone(),
        { x: sx(data.elbow), y: sy(data.elbow) },
        { x: sx(data.wrist), y: sy(data.wrist) },
        { x: sx(data.shoulder), y: sy(data.shoulder) },
        upperJoint.length,
        foreJoint.length,
        armState,
        config,
      );
      flips += arm.flips;
      elbowClamped += arm.clamped;
      armState.elbowZ = arm.upperDir.z * upperJoint.length;
      armState.wristZ = arm.foreDir.z * foreJoint.length;

      const parentWorld = world.get('chest') ?? new Quaternion();
      const upperRest = upperJoint.dir.clone().applyQuaternion(parentWorld);
      const qUpper = new Quaternion().setFromUnitVectors(upperRest, safeNormalize(arm.upperDir.clone()));
      const wUpper = qUpper.multiply(parentWorld.clone());

      const lowerRest = foreJoint.dir.clone().applyQuaternion(wUpper);
      const qLower = new Quaternion().setFromUnitVectors(lowerRest, safeNormalize(arm.foreDir.clone()));
      const wLower = qLower.multiply(wUpper.clone());

      const handSolution = solveHand(
        frame,
        data.hand,
        dataSide,
        side,
        arm.foreDir,
        skeleton,
        config,
        this.state.hands[side],
        xSign,
      );
      const wHand = handSolution.handQuat.clone();
      if (handSolution.flipped) {
        flips++;
        handFlips++;
      }
      handConfidence = Math.min(handConfidence, handSolution.confidence);

      setWorld(bones[0], wUpper);
      setWorld(bones[1], wLower);
      setWorld(bones[2], wHand);

      for (const finger of FINGERS) {
        const chain = fingerBoneNames(side, finger);
        let parent = wHand;
        for (const bone of chain) {
          const target = handSolution.fingerDirs.get(bone);
          const restDir = skeleton.joints.get(bone)?.dir;
          if (!target || !restDir) {
            setWorld(bone, parent.clone());
            continue;
          }
          const currentRest = restDir.clone().applyQuaternion(parent);
          const delta = new Quaternion().setFromUnitVectors(currentRest, safeNormalize(target.clone()));
          const w = delta.multiply(parent.clone());
          setWorld(bone, w);
          parent = w;
        }
      }
    }

    // ---- head ------------------------------------------------------------
    const neckWorld = world.get('neck') ?? new Quaternion();
    const fitted = solveHead(frame, config, xSign);
    const usable = fitted.success && fitted.residual < 0.16;
    const headPose = usable ? fitted : fallbackHeadPose(frame, xSign);
    this.state.headResidual = fitted.residual;
    const headLocal = new Quaternion().slerp(
      headPose.quat,
      usable ? clamp(config.head.gain, 0, 1) : 0.35,
    );
    setWorld('head', neckWorld.clone().multiply(headLocal));

    // ---- local rotations, smoothing, and face ----------------------------
    const locals = new Map<string, Quaternion>();
    const bodyAlpha = halfLifeAlpha(dt, config.smoothing.bodyHalfLife);
    const handAlpha = halfLifeAlpha(dt, config.smoothing.handHalfLife);
    for (const name of skeleton.order) {
      const joint = skeleton.joints.get(name)!;
      const parentWorld = joint.parent ? world.get(joint.parent) ?? new Quaternion() : new Quaternion();
      const targetWorld = world.get(name) ?? parentWorld.clone();
      const targetLocal = parentWorld.clone().invert().multiply(targetWorld);
      const isHand = this.fingerSet.has(name) || name.endsWith('Hand');
      const alpha = isHand ? handAlpha : bodyAlpha;
      const prev = this.previousLocals.get(name);
      const local = prev ? prev.clone().slerp(targetLocal, alpha) : targetLocal.clone();
      local.normalize();
      locals.set(name, local);
      this.previousLocals.set(name, local.clone());
    }

    const rawFace = extractFaceChannels(frame);
    this.calibrator.add(rawFace);
    const calibrated = this.calibrator.apply(rawFace);
    const face = this.smoothFace(calibrated, dt);
    if (mirror) {
      face.gazeX = -face.gazeX;
    }

    const meta: PoseMeta = {
      depthFlips: flips,
      handConfidence,
      elbowClamped,
      headFitResidual: fitted.residual,
    };
    this.state.flips += flips;
    this.state.handFlips += handFlips;
    this.state.elbowClamped += elbowClamped;
    this.state.handConfidence = handConfidence;

    return { locals, face, meta };
  }

  /** Arm solutions from the last frame (debug/overlay use). */

  private smoothFace(input: FaceChannels, dt: number): FaceChannels {
    const a = halfLifeAlpha(dt, this.config.face.smoothingHalfLife);
    const prev = this.previousFace;
    const out: FaceChannels = {
      browRaise: prev.browRaise + (input.browRaise - prev.browRaise) * a,
      browFurrow: prev.browFurrow + (input.browFurrow - prev.browFurrow) * a,
      eyeOpen: prev.eyeOpen + (input.eyeOpen - prev.eyeOpen) * a,
      jawOpen: prev.jawOpen + (input.jawOpen - prev.jawOpen) * a,
      mouthWide: prev.mouthWide + (input.mouthWide - prev.mouthWide) * a,
      mouthRound: prev.mouthRound + (input.mouthRound - prev.mouthRound) * a,
      smile: prev.smile + (input.smile - prev.smile) * a,
      frown: prev.frown + (input.frown - prev.frown) * a,
      gazeX: prev.gazeX + (input.gazeX - prev.gazeX) * a,
      gazeY: prev.gazeY + (input.gazeY - prev.gazeY) * a,
    };
    this.previousFace = out;
    return out;
  }
}

/** Rest-pose forward kinematics for evaluation and keypoint overlays. */
export function forwardKinematics(
  skeleton: Skeleton,
  locals: Map<string, Quaternion>,
): { positions: Map<string, Vector3>; quats: Map<string, Quaternion> } {
  const positions = new Map<string, Vector3>();
  const quats = new Map<string, Quaternion>();
  for (const name of skeleton.order) {
    const joint = skeleton.joints.get(name)!;
    const local = locals.get(name) ?? new Quaternion();
    if (!joint.parent) {
      positions.set(name, joint.pos.clone());
      quats.set(name, local.clone());
      continue;
    }
    const parentPos = positions.get(joint.parent) ?? joint.pos;
    const parentQuat = quats.get(joint.parent) ?? new Quaternion();
    const restOffset = joint.pos.clone().sub(skeleton.joints.get(joint.parent)!.pos);
    const pos = parentPos.clone().add(restOffset.applyQuaternion(parentQuat));
    positions.set(name, pos);
    quats.set(name, parentQuat.clone().multiply(local));
  }
  return { positions, quats };
}
