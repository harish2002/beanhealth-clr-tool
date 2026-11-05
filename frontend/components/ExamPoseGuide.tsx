"use client";

/**
 * BeanHealth CLR — position guide for the gaze exam
 * =================================================
 *
 * For the position being captured: a drawing of the patient as the camera
 * sees them (head turn, chin, tilt, covered eye, where the eyes look), and a
 * live map of the head pose with the position's accepted window. Drawn from
 * the capture plan, so it always matches what frame QC will accept.
 */

import { useId } from "react";
import { figureCaption, figureFor, screenPose } from "@/lib/examFigure";
import { poseError } from "@/lib/examGuidance";
import type { ExamHeadPose, ExamPositionPlan } from "@/lib/types";

const SKIN = "#EAD3BE";
const SKIN_EDGE = "#C9A98D";
const HAIR = "#4B3C33";
const HAND = "#DDBB9D";
const IRIS = "#6B4B30";
const LIGHT = "#F59E0B";
const ACCENT = "#1B47C4";            // clinical-600

// Eye centres in the drawing: the image-left eye is the patient's RIGHT eye.
const EYE_X = { R: 78, L: 122 } as const;
const EYE_Y = 116;

function FaceFigure({ plan, distanceCm }: { plan: ExamPositionPlan; distanceCm: number }) {
  const f = figureFor(plan, distanceCm);
  const id = useId().replace(/:/g, "");
  const fx = f.featureDx * 46, fy = f.featureDy * 40;
  // Card targets sit to the patient's right (image left) and left (image right) of the light.
  const targets = [
    { key: "patient_right", x: 58 },
    { key: "light", x: 100 },
    { key: "patient_left", x: 142 },
  ] as const;
  return (
    <svg viewBox="0 0 200 232" className="w-[132px] lg:w-[230px] h-auto" role="img"
         aria-label={`Target position: ${figureCaption(plan)}`}>
      {/* The phone: the light at the lens and the two calibration dots */}
      <rect x="34" y="6" width="132" height="20" rx="6" fill="#1F2937" />
      {targets.map((t) => {
        const on = t.key === f.target;
        return t.key === "light" ? (
          <g key={t.key}>
            <circle cx={t.x} cy="16" r="8" fill={LIGHT} opacity={on ? 0.35 : 0.15} />
            <circle cx={t.x} cy="16" r="4" fill={LIGHT} />
            {on && <path d={`M${t.x - 5} 34 L${t.x} 29 L${t.x + 5} 34`} fill="none" stroke={ACCENT} strokeWidth="2" />}
          </g>
        ) : (
          <g key={t.key}>
            <circle cx={t.x} cy="16" r="3.2" fill={on ? "#fff" : "#6B7280"} />
            {on && <circle cx={t.x} cy="16" r="6.5" fill="none" stroke={ACCENT} strokeWidth="2" />}
            {on && <path d={`M${t.x - 5} 34 L${t.x} 29 L${t.x + 5} 34`} fill="none" stroke={ACCENT} strokeWidth="2" />}
          </g>
        );
      })}

      {/* The neck stays put; a tilt turns the head on it. */}
      <rect x="86" y="190" width="28" height="34" fill={SKIN} stroke={SKIN_EDGE} />
      <g transform={`rotate(${f.rotateDeg} 100 140)`}>
        <ellipse cx={44 + fx * 0.25} cy="132" rx="7" ry="13" fill={SKIN} stroke={SKIN_EDGE} />
        <ellipse cx={156 + fx * 0.25} cy="132" rx="7" ry="13" fill={SKIN} stroke={SKIN_EDGE} />
        <ellipse cx="100" cy="128" rx="56" ry="70" fill={SKIN} stroke={SKIN_EDGE} strokeWidth="1.5" />
        <path d="M44 122 Q45 58 100 56 Q155 58 156 122 Q150 84 100 80 Q50 84 44 122 Z" fill={HAIR} />

        <g transform={`translate(${fx} ${fy})`}>
          {(["R", "L"] as const).map((eye) => {
            const ex = EYE_X[eye];
            const covered = f.coveredEye === eye;
            return (
              <g key={eye}>
                <path d={`M${ex - 13} ${EYE_Y - 13} Q${ex} ${EYE_Y - 19} ${ex + 13} ${EYE_Y - 13}`} fill="none" stroke={HAIR} strokeWidth="2.5" strokeLinecap="round" />
                <clipPath id={`${id}${eye}`}>
                  <ellipse cx={ex} cy={EYE_Y} rx="13" ry="7.5" />
                </clipPath>
                <ellipse cx={ex} cy={EYE_Y} rx="13" ry="7.5" fill="#fff" stroke={SKIN_EDGE} />
                <g clipPath={`url(#${id}${eye})`}>
                  <circle cx={ex + f.gazeDx * 7} cy={EYE_Y + f.gazeDy * 4} r="6.2" fill={IRIS} />
                  <circle cx={ex + f.gazeDx * 7} cy={EYE_Y + f.gazeDy * 4} r="2.7" fill="#111" />
                  {/* The torch's corneal reflex */}
                  <circle cx={ex + f.gazeDx * 7 + 1.2} cy={EYE_Y + f.gazeDy * 4 - 1.2} r="1.2" fill="#fff" />
                </g>
                {covered && (
                  <g>
                    <rect x={ex - 19} y={EYE_Y - 17} width="38" height="32" rx="13" fill={HAND} stroke="#A9825F" strokeWidth="1.2" />
                    <path d={`M${ex - 9} ${EYE_Y - 15} v27 M${ex} ${EYE_Y - 16} v29 M${ex + 9} ${EYE_Y - 15} v27`} stroke="#A9825F" strokeWidth="0.9" opacity="0.6" />
                  </g>
                )}
              </g>
            );
          })}
          <path d={`M100 124 Q${97 + fx * 0.15} 142 ${93 + fx * 0.2} 147 Q${100 + fx * 0.2} 151 ${107 + fx * 0.2} 147`}
                fill="none" stroke={SKIN_EDGE} strokeWidth="1.6" strokeLinecap="round" />
          <path d="M87 167 Q100 175 113 167" fill="none" stroke="#B2766A" strokeWidth="2.2" strokeLinecap="round" />
        </g>
      </g>
    </svg>
  );
}

const MAP = 150, C = MAP / 2, PX_PER_DEG = 1.55;   // ±40° across the map
const at = (deg: number) => C + Math.max(-45, Math.min(45, deg)) * PX_PER_DEG;

function PoseMap({ plan, pose }: { plan: ExamPositionPlan; pose: ExamHeadPose | null }) {
  const target = screenPose(plan.head);
  const tol = plan.head.tolerance_deg * PX_PER_DEG;
  const live = pose ? screenPose(pose) : null;
  const ok = poseError(pose, plan) === 0;
  const roll = plan.head.roll_deg;
  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox={`0 0 ${MAP} ${MAP}`} className="w-[120px] lg:w-[200px] h-auto" role="img" aria-label="Head position: target window and current pose">
        <rect x="0.5" y="0.5" width={MAP - 1} height={MAP - 1} rx="8" fill="#F8FAFC" stroke="#E2E8F0" />
        <line x1={C} y1="16" x2={C} y2={MAP - 16} stroke="#E2E8F0" />
        <line x1="16" y1={C} x2={MAP - 16} y2={C} stroke="#E2E8F0" />
        <rect x={at(target.x) - tol} y={at(target.y) - tol} width={2 * tol} height={2 * tol} rx="3"
              fill={ok ? "#D1FAE5" : "#DCE5FB"} stroke={ok ? "#059669" : ACCENT} strokeWidth="1.5" />
        {live && (
          <circle cx={at(live.x)} cy={at(live.y)} r="5.5" fill={ok ? "#059669" : "#D97706"} stroke="#fff" strokeWidth="1.5" />
        )}
        <text x={C} y="11" textAnchor="middle" fontSize="8.5" fill="#64748B">chin up</text>
        <text x={C} y={MAP - 5} textAnchor="middle" fontSize="8.5" fill="#64748B">chin down</text>
        <text x="9" y={C} textAnchor="middle" fontSize="8.5" fill="#64748B" transform={`rotate(-90 9 ${C})`}>their right</text>
        <text x={MAP - 9} y={C} textAnchor="middle" fontSize="8.5" fill="#64748B" transform={`rotate(90 ${MAP - 9} ${C})`}>their left</text>
      </svg>
      {roll && (
        <svg viewBox="0 0 150 30" className="w-[120px] lg:w-[200px] h-auto" role="img" aria-label="Head tilt: target range and current tilt">
          <line x1="8" y1="12" x2="142" y2="12" stroke="#E2E8F0" strokeWidth="6" strokeLinecap="round" />
          {/* Screen x = −roll: a tilt to the patient's right shoulder leans to the image left. */}
          <line x1={75 - roll.max * 1.1} y1="12" x2={75 - roll.min * 1.1} y2="12" stroke={ok ? "#059669" : ACCENT} strokeWidth="6" strokeLinecap="round" />
          {pose && <circle cx={75 - Math.max(-60, Math.min(60, pose.roll_deg)) * 1.1} cy="12" r="5" fill={ok ? "#059669" : "#D97706"} stroke="#fff" strokeWidth="1.5" />}
          <text x="8" y="28" fontSize="8.5" fill="#64748B">right shoulder</text>
          <text x="142" y="28" textAnchor="end" fontSize="8.5" fill="#64748B">left shoulder</text>
        </svg>
      )}
      <p className="text-[11px] text-ink-400">{live ? (ok ? "In the window — hold still" : "Move the dot into the box") : "Waiting for the face"}</p>
    </div>
  );
}

export default function ExamPoseGuide({ plan, pose, distanceCm, className = "" }: {
  plan: ExamPositionPlan; pose: ExamHeadPose | null; distanceCm: number; className?: string;
}) {
  return (
    <div className={`surface p-3 lg:p-4 ${className}`}>
      <div className="flex flex-wrap items-start justify-center gap-x-3 lg:gap-x-6 gap-y-3">
        <FaceFigure plan={plan} distanceCm={distanceCm} />
        <PoseMap plan={plan} pose={pose} />
      </div>
      <p className="mt-2 text-[13px] text-ink-800 text-center">{figureCaption(plan)}</p>
      <p className="text-[11px] text-ink-400 text-center">
        Drawn as the camera sees the patient: their right is on your left.
      </p>
    </div>
  );
}
