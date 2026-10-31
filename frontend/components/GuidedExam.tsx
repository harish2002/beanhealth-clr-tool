"use client";

/**
 * BeanHealth CLR — Guided gaze examination (Phase 3 capture)
 * ==========================================================
 *
 * Walks the operator through the capture plan the server hands out
 * (GET /exam/protocol): calibration with one eye covered, then each gaze
 * position reached by turning the patient's head, then the head tilts.
 *
 *  1. Rear camera (torch on where supported) and MediaPipe Face Mesh.
 *  2. Every frame: head pose from the mesh → live guidance toward the
 *     position's target ("turn the head 10° more to the patient's left").
 *  3. While recording, every frame's landmarks are extracted on the phone
 *     (lib/examLandmarks — limbus, reflex, pupil, lids, sharpness) for the
 *     position's hold time. Only these numbers are kept; no image is stored
 *     or uploaded.
 *  4. On the operator's request the landmark tracks go to /exam/evaluate:
 *     frame QC, calibration, the per-gaze grid (PD ± error) and the rules
 *     engine's findings and referral decision. Positions without enough
 *     valid frames can be retaken.
 *
 * Decision support — not a diagnosis. The clinician report is the next
 * milestone; this screen shows the grid and findings to the operator.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FaceMesh, Results } from "@mediapipe/face_mesh";
import { getExamProtocol, postExamEvaluate, postExamReport } from "@/lib/api";
import { rgbaToGray } from "@/lib/eyeVision";
import { eyeObservation, eyeRegionHalf, headPose, irisGuess } from "@/lib/examLandmarks";
import type { GrayRegion } from "@/lib/examLandmarks";
import { buildSession, deviceModel, effectiveFps, poseGuidance } from "@/lib/examGuidance";
import {
  findingsToShow, gazeName, gridLayout, horizontalKind, REFERRAL_TITLE, sizeParts, verticalKind, verticalLabel,
} from "@/lib/examResults";
import type {
  ExamEye,
  ExamEyeObservation,
  ExamFrameObservation,
  ExamHeadPose,
  ExamMeasured,
  ExamPositionPlan,
  ExamProtocol,
  ExamCaptureSession,
  ExamProtocolPlan,
  ExamResult,
} from "@/lib/types";

type Stage = "setup" | "capture" | "review" | "uploading" | "results";
type V3 = [number, number, number];

interface Live {
  pose: ExamHeadPose | null;
  eyes: Partial<Record<ExamEye, ExamEyeObservation>>;
  fps: number | null;
}

interface Recording { position: string; start: number; frames: ExamFrameObservation[] }

const EYE_NAME: Record<ExamEye, string> = { R: "Right eye", L: "Left eye" };
const LIVE_EYES_EVERY = 6;             // between recordings, refresh the eye read-out every 6th frame
const UI_EVERY_MS = 150;
// A slow phone may process only a few frames a second: keep recording past the
// hold time until there are this many × the position's frame minimum, but
// never longer than MAX_HOLD_FACTOR × the hold time.
const FRAME_MARGIN = 1.5;
const MAX_HOLD_FACTOR = 4;

/** Whether a recording has enough: the hold time and enough frames (or the time cap). */
function recordingDone(p: ExamPositionPlan, elapsedMs: number, frames: number): boolean {
  const hold = p.hold_s * 1000;
  return (elapsedMs >= hold && frames >= Math.ceil(FRAME_MARGIN * p.min_valid_frames))
    || elapsedMs >= MAX_HOLD_FACTOR * hold;
}

function observedEyes(p: ExamPositionPlan): ExamEye[] {
  return p.fixing_eye ? [p.fixing_eye] : ["R", "L"];
}

function positionLabel(p: ExamPositionPlan): string {
  if (p.kind === "calibration") {
    const eye = p.fixing_eye === "R" ? "right" : "left";
    const target = p.target_offset_cm === 0 ? "light" : `target ${p.target_offset_cm > 0 ? "right" : "left"}`;
    return `Calibrate ${eye} eye · ${target}`;
  }
  return p.position.replace("_", " ").replace(/^\w/, (c) => c.toUpperCase());
}

export default function GuidedExam() {
  const [stage, setStage] = useState<Stage>("setup");
  const [protocol, setProtocol] = useState<ExamProtocol>("nine_gaze");
  const [tilt, setTilt] = useState(true);
  const [distanceCm, setDistanceCm] = useState("36");
  const [plan, setPlan] = useState<ExamProtocolPlan | null>(null);
  const [idx, setIdx] = useState(0);
  const [captures, setCaptures] = useState<Record<string, ExamFrameObservation[]>>({});
  const [live, setLive] = useState<Live>({ pose: null, eyes: {}, fps: null });
  const [recProgress, setRecProgress] = useState<number | null>(null);
  const [torch, setTorch] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExamResult | null>(null);
  const uploadedRef = useRef<ExamCaptureSession | null>(null);
  const qc = result?.qc ?? null;

  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLCanvasElement | null>(null);
  const fmRef = useRef<FaceMesh | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);
  const recRef = useRef<Recording | null>(null);
  const capturesRef = useRef<Record<string, ExamFrameObservation[]>>({});
  const tickRef = useRef({ n: 0, lastUi: 0, times: [] as number[] });
  const sessionRef = useRef({ id: "", startedAt: "", width: 0, height: 0 });
  const planRef = useRef<ExamProtocolPlan | null>(null);
  const idxRef = useRef(0);
  planRef.current = plan;
  idxRef.current = idx;

  const current = plan?.positions[idx] ?? null;
  const guidance = useMemo(() => (current ? poseGuidance(live.pose, current) : null), [current, live.pose]);

  // ── Frame processing ────────────────────────────────────────────────────

  const drawOverlay = useCallback((eyes: Partial<Record<ExamEye, ExamEyeObservation>>, W: number, H: number) => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const cw = video.clientWidth, ch = video.clientHeight;
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, cw, ch);
    // object-contain: the frame is letterboxed inside the element.
    const s = Math.min(cw / W, ch / H), ox = (cw - W * s) / 2, oy = (ch - H * s) / 2;
    for (const obs of Object.values(eyes)) {
      if (!obs) continue;
      if (obs.limbus) {
        const l = obs.limbus;
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.ellipse(ox + l.centre_px[0] * s, oy + l.centre_px[1] * s, l.semi_major_px * s, l.semi_minor_px * s,
                    (l.angle_deg * Math.PI) / 180, 0, 2 * Math.PI);
        ctx.stroke();
      }
      if (obs.reflex_px) {
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath();
        ctx.arc(ox + obs.reflex_px[0] * s, oy + obs.reflex_px[1] * s, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }, []);

  const processResults = useCallback((r: Results) => {
    const frame = frameRef.current;
    const p = planRef.current?.positions[idxRef.current];
    if (!frame || !p) return;
    const now = performance.now();
    const W = frame.width, H = frame.height;
    const t = tickRef.current;
    t.n++;
    t.times.push(now);
    while (t.times.length > 30) t.times.shift();

    const lms = r.multiFaceLandmarks?.[0];
    const points: V3[] | null = lms && lms.length >= 478 ? lms.map((q) => [q.x * W, q.y * H, q.z * W]) : null;
    const pose = points ? headPose(points) : null;
    const rec = recRef.current;
    const recording = rec !== null && rec.position === p.position;

    let eyes: Partial<Record<ExamEye, ExamEyeObservation>> | null = null;
    let sharp: number | null = null;
    if (points && (recording || t.n % LIVE_EYES_EVERY === 0)) {
      const ctx = frame.getContext("2d", { willReadFrequently: true });
      const canonical = planRef.current?.settings.sharpness_canonical_iris_radius_px ?? 32;
      eyes = {};
      const sharps: (number | null)[] = [];
      for (const eye of observedEyes(p)) {
        const g = irisGuess(points, eye);
        const half = eyeRegionHalf(g.radius);
        const x0 = Math.max(0, Math.floor(g.centre[0]) - half), y0 = Math.max(0, Math.floor(g.centre[1]) - half);
        const x1 = Math.min(W, Math.floor(g.centre[0]) + half + 1), y1 = Math.min(H, Math.floor(g.centre[1]) + half + 1);
        if (!ctx || x1 <= x0 || y1 <= y0) { sharps.push(null); continue; }
        const img = ctx.getImageData(x0, y0, x1 - x0, y1 - y0);
        const region: GrayRegion = { data: rgbaToGray(img.data), w: x1 - x0, h: y1 - y0, x0, y0, fullW: W, fullH: H };
        const res = eyeObservation(region, points, eye, canonical);
        eyes[eye] = res.obs;
        sharps.push(res.sharpness);
      }
      // The blurrier eye decides; a missing value stays missing (QC rejects it).
      sharp = sharps.length && sharps.every((v) => v !== null) ? Math.min(...(sharps as number[])) : null;
    }

    if (recording && rec) {
      const tMs = now - rec.start;
      rec.frames.push(points && eyes
        ? { index: rec.frames.length, t_ms: tMs, head_pose: pose, sharpness: sharp, eyes }
        : { index: rec.frames.length, t_ms: tMs, head_pose: null, sharpness: null, eyes: {} });
      if (recordingDone(p, tMs, rec.frames.length)) {
        recRef.current = null;
        capturesRef.current = { ...capturesRef.current, [p.position]: rec.frames };
        setCaptures(capturesRef.current);
        setRecProgress(null);
        const n = planRef.current?.positions.length ?? 0;
        const next = planRef.current?.positions.findIndex((q, i) => i > idxRef.current && !capturesRef.current[q.position]?.length) ?? -1;
        if (next >= 0) setIdx(next);
        else { setIdx(Math.min(idxRef.current, n - 1)); setStage("review"); }
      } else if (now - t.lastUi > UI_EVERY_MS) {
        setRecProgress(tMs / (p.hold_s * 1000));
      }
    }

    if (eyes) drawOverlay(eyes, W, H);
    if (now - t.lastUi > UI_EVERY_MS) {
      t.lastUi = now;
      const span = t.times.length > 1 ? t.times[t.times.length - 1] - t.times[0] : 0;
      setLive((prev) => ({ pose, eyes: eyes ?? prev.eyes, fps: span > 0 ? ((t.times.length - 1) * 1000) / span : null }));
    }
  }, [drawOverlay]);

  const processRef = useRef(processResults);
  processRef.current = processResults;

  // ── MediaPipe and the frame loop ────────────────────────────────────────

  const cameraOn = stage === "capture";

  useEffect(() => {
    if (!cameraOn || fmRef.current) return;
    let cancelled = false;
    (async () => {
      try {
        const { FaceMesh: FaceMeshCtor } = await import("@mediapipe/face_mesh");
        if (cancelled) return;
        const fm = new FaceMeshCtor({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`,
        });
        fm.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        fm.onResults((r) => processRef.current(r));
        fmRef.current = fm;
      } catch (e) {
        console.warn("[GuidedExam] Face Mesh failed to load:", e);
        setError("The face-tracking model could not be loaded. Check your connection and reload.");
      }
    })();
    return () => { cancelled = true; };
  }, [cameraOn]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    fmRef.current?.close().catch(() => undefined);
    fmRef.current = null;
  }, []);

  useEffect(() => {
    if (!cameraOn) return;
    let alive = true;
    let raf = 0;
    const tick = async () => {
      if (!alive) return;
      const video = videoRef.current;
      const fm = fmRef.current;
      if (video && fm && video.readyState >= 2 && video.videoWidth && !busyRef.current) {
        if (!frameRef.current) frameRef.current = document.createElement("canvas");
        const frame = frameRef.current;
        if (frame.width !== video.videoWidth || frame.height !== video.videoHeight) {
          frame.width = video.videoWidth;
          frame.height = video.videoHeight;
          sessionRef.current.width = frame.width;
          sessionRef.current.height = frame.height;
        }
        frame.getContext("2d", { willReadFrequently: true })?.drawImage(video, 0, 0);
        busyRef.current = true;
        try {
          await fm.send({ image: frame });
        } catch (e) {
          console.warn("[GuidedExam] frame failed:", e);
        } finally {
          busyRef.current = false;
        }
      }
      if (alive) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [cameraOn]);

  // ── Camera ──────────────────────────────────────────────────────────────

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 },
                 frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const track = stream.getVideoTracks()[0];
      const caps = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & { torch?: boolean };
      if (caps.torch) {
        try {
          await track.applyConstraints({ advanced: [{ torch: true } as MediaTrackConstraintSet] });
          setTorch(true);
        } catch {
          setTorch(false);
        }
      } else {
        setTorch(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Camera access was denied.");
    }
  }, []);

  useEffect(() => {
    if (cameraOn && !streamRef.current) void startCamera();
    if (!cameraOn && streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, [cameraOn, startCamera]);

  const toggleTorch = useCallback(async () => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track || torch === null) return;
    try {
      await track.applyConstraints({ advanced: [{ torch: !torch } as MediaTrackConstraintSet] });
      setTorch(!torch);
    } catch {
      setTorch(null);
    }
  }, [torch]);

  // ── Flow ────────────────────────────────────────────────────────────────

  const begin = useCallback(async () => {
    setError(null);
    try {
      const p = await getExamProtocol(protocol, tilt);
      setPlan(p);
      setIdx(0);
      capturesRef.current = {};
      setCaptures({});
      setResult(null);
      sessionRef.current = { ...sessionRef.current, id: crypto.randomUUID(), startedAt: new Date().toISOString() };
      setStage("capture");
    } catch (e) {
      setError(e instanceof Error && e.message === "NETWORK_ERROR"
        ? "The server could not be reached. Check the connection and try again."
        : "The capture plan could not be loaded.");
    }
  }, [protocol, tilt]);

  const record = useCallback(() => {
    if (!current) return;
    recRef.current = { position: current.position, start: performance.now(), frames: [] };
    setRecProgress(0);
  }, [current]);

  const retake = useCallback((position: string) => {
    const i = plan?.positions.findIndex((p) => p.position === position) ?? -1;
    if (i < 0) return;
    setIdx(i);
    setStage("capture");
  }, [plan]);

  const upload = useCallback(async () => {
    if (!plan) return;
    const d = Number(distanceCm);
    const session = buildSession({
      sessionId: sessionRef.current.id, plan, captures: capturesRef.current,
      frameSize: { width: sessionRef.current.width, height: sessionRef.current.height },
      distanceCm: Number.isFinite(d) && d > 0 ? d : null, distanceErrorCm: 1,
      deviceModel: deviceModel(navigator.userAgent), startedAt: sessionRef.current.startedAt,
    });
    setStage("uploading");
    setError(null);
    try {
      setResult(await postExamEvaluate(session, tilt));
      uploadedRef.current = session;
      setStage("results");
    } catch (e) {
      setError(e instanceof Error && e.message === "TIMEOUT" ? "The server took too long. Try again."
        : "The session could not be checked. Check the connection and try again.");
      setStage("review");
    }
  }, [plan, distanceCm, tilt]);

  /** Open the clinician report in a new tab (opened before the request, so pop-up blockers allow it). */
  const openReport = useCallback(async (format: "pdf" | "html", label: string, age: string): Promise<string | null> => {
    const session = uploadedRef.current;
    if (!session) return "Upload the session first.";
    const tab = window.open("", "_blank");
    const ageYears = Number(age);
    try {
      const blob = await postExamReport(session, format, {
        label: label.trim() || null, age_years: age.trim() && Number.isFinite(ageYears) ? Math.round(ageYears) : null,
      }, tilt);
      const url = URL.createObjectURL(blob);
      if (tab) tab.location.href = url;
      else window.location.href = url;
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
      return null;
    } catch (e) {
      tab?.close();
      return e instanceof Error && e.message === "TIMEOUT" ? "The report took too long. Try again."
        : "The report could not be made. Check the connection and try again.";
    }
  }, [tilt]);

  // ── Render ──────────────────────────────────────────────────────────────

  if (stage === "setup") {
    return (
      <div className="surface-raised p-6 max-w-xl space-y-5">
        <div>
          <p className="eyebrow">Gaze examination</p>
          <h2 className="display-md mt-1">Set up the capture</h2>
          <p className="lede text-[14px] mt-2">
            Phone on a stand, rear camera facing the patient at eye level, torch on. The patient keeps
            looking at the light while you turn their head to each position.
          </p>
        </div>
        <label className="block text-[13px] text-ink-700">
          Protocol
          <select className="mt-1 block w-full border border-ink-200 rounded-md px-3 py-2 bg-white"
                  value={protocol} onChange={(e) => setProtocol(e.target.value as ExamProtocol)}>
            <option value="nine_gaze">9 gaze positions</option>
            <option value="five_gaze">Kids mode · 5 positions</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-[13px] text-ink-700">
          <input type="checkbox" checked={tilt} onChange={(e) => setTilt(e.target.checked)} />
          Include the head-tilt captures
        </label>
        <label className="block text-[13px] text-ink-700">
          Camera-to-eye distance (cm, measured)
          <input type="number" inputMode="decimal" min={20} max={60} step={0.5}
                 className="mt-1 block w-full border border-ink-200 rounded-md px-3 py-2"
                 value={distanceCm} onChange={(e) => setDistanceCm(e.target.value)} />
          <span className="text-[11.5px] text-ink-400">
            33–40 cm. Calibration turns the card targets into angles with this distance.
          </span>
        </label>
        {error && <p className="text-[13px] text-red-600">{error}</p>}
        <button className="btn-primary" onClick={begin}>Start the camera</button>
        <p className="text-[11.5px] text-ink-400">
          Only measurements leave the phone — never the video. Decision support, not a diagnosis.
        </p>
      </div>
    );
  }

  const done = plan ? plan.positions.filter((p) => capturesRef.current[p.position]?.length).length : 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
      <div className="space-y-3">
        <div className={`relative bg-black rounded-card overflow-hidden aspect-video ${cameraOn ? "" : "hidden"}`}>
          <video ref={videoRef} playsInline muted className="w-full h-full object-contain" />
          <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
          {recProgress !== null && (
            <div className="absolute bottom-0 left-0 h-1.5 bg-emerald-500" style={{ width: `${Math.min(100, recProgress * 100)}%` }} />
          )}
        </div>

        {stage === "results" && result && <Results result={result} plan={plan} onRetake={retake} onReport={openReport} />}

        {(stage === "review" || stage === "uploading") && (
          <div className="surface-raised p-5 space-y-3">
            <h3 className="display-md text-[20px]">{done} of {plan?.positions.length} positions recorded</h3>
            <p className="lede text-[13.5px]">
              Upload the measurements to check every frame, fit the calibration and measure each gaze. Only landmark
              numbers are sent, never images — and nothing until you press the button.
            </p>
            {error && <p className="text-[13px] text-red-600">{error}</p>}
            <div className="flex gap-2">
              <button className="btn-primary" onClick={upload} disabled={stage === "uploading" || done === 0}>
                {stage === "uploading" ? "Measuring…" : "Measure and evaluate"}
              </button>
              <button className="btn-secondary" onClick={() => setStage("capture")}>Back to capture</button>
            </div>
          </div>
        )}
      </div>

      <aside className="space-y-4">
        {stage === "capture" && current && (
          <div className="surface-raised p-5 space-y-3">
            <p className="eyebrow">Position {idx + 1} of {plan?.positions.length}</p>
            <h3 className="text-[17px] font-semibold text-ink-900">{positionLabel(current)}</h3>
            <p className="text-[13.5px] text-ink-600">{current.instruction}</p>
            <div className={`rounded-md px-3 py-2 text-[13px] ${guidance?.onTarget ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-800"}`}>
              {guidance?.onTarget ? "On target — hold still" : guidance?.hints.map((h) => <div key={h}>{h}</div>)}
            </div>
            <dl className="grid grid-cols-3 gap-1 text-[12px] text-ink-500 font-mono">
              <dt>yaw</dt><dt>pitch</dt><dt>roll</dt>
              <dd className="text-ink-900">{live.pose ? live.pose.yaw_deg.toFixed(0) : "–"}°</dd>
              <dd className="text-ink-900">{live.pose ? live.pose.pitch_deg.toFixed(0) : "–"}°</dd>
              <dd className="text-ink-900">{live.pose ? live.pose.roll_deg.toFixed(0) : "–"}°</dd>
            </dl>
            <ul className="text-[12.5px] space-y-1">
              {observedEyes(current).map((eye) => {
                const o = live.eyes[eye];
                return (
                  <li key={eye} className="flex justify-between">
                    <span className="text-ink-600">{EYE_NAME[eye]}</span>
                    <span className={o?.limbus && o.reflex_px ? "text-emerald-600" : "text-amber-700"}>
                      {!o ? "–" : !o.limbus ? "limbus not found" : !o.reflex_px ? "no reflex" : "limbus · reflex"}
                    </span>
                  </li>
                );
              })}
            </ul>
            <p className="text-[11.5px] text-ink-400">{live.fps ? `${live.fps.toFixed(0)} frames/s processed` : "Starting…"}</p>
            {error && <p className="text-[13px] text-red-600">{error}</p>}
            <div className="flex flex-wrap gap-2">
              <button className="btn-primary" onClick={record} disabled={recProgress !== null || !live.pose}>
                {recProgress !== null ? "Recording…" : capturesRef.current[current.position]?.length ? "Record again" : `Record ${current.hold_s} s`}
              </button>
              <button className="btn-secondary" disabled={recProgress !== null}
                      onClick={() => (idx + 1 < (plan?.positions.length ?? 0) ? setIdx(idx + 1) : setStage("review"))}>
                Skip
              </button>
              <button className="btn-secondary" disabled={recProgress !== null || done === 0} onClick={() => setStage("review")}>
                Finish
              </button>
              <button className="btn-secondary" disabled={torch === null} onClick={toggleTorch}>
                {torch === null ? "Torch unavailable" : torch ? "Torch on" : "Torch off"}
              </button>
            </div>
          </div>
        )}

        {plan && (
          <ol className="surface p-4 text-[12.5px] space-y-1">
            {plan.positions.map((p, i) => {
              const frames = captures[p.position];
              const fps = frames ? effectiveFps(frames) : null;
              const status = qc?.positions.find((q) => q.position === p.position);
              return (
                <li key={p.position} className={`flex justify-between gap-2 ${i === idx && stage === "capture" ? "font-semibold text-ink-900" : "text-ink-600"}`}>
                  <button className="text-left truncate hover:underline" onClick={() => { setIdx(i); setStage("capture"); }}>
                    {positionLabel(p)}
                  </button>
                  <span className={status ? (status.status === "OK" ? "text-emerald-600" : "text-red-600") : "text-ink-400"}>
                    {status ? `${status.frames_valid}/${status.frames_total}` : frames?.length ? `${frames.length} fr${fps ? ` · ${fps.toFixed(0)}/s` : ""}` : "–"}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </aside>
    </div>
  );
}

function fmt(m: { value: number; error: number }, digits = 2): string {
  return `${m.value.toFixed(digits)} ± ${m.error.toFixed(digits)}`;
}

function CellValue({ m, kind }: { m: ExamMeasured; kind: string }) {
  const [v, e] = sizeParts(m);
  return (
    <div>
      <span className="font-mono text-[13px] text-ink-900">{v}</span>
      <span className="font-mono text-ink-400"> ±{e}</span>
      <div className="text-ink-600">{kind}</div>
    </div>
  );
}

const DECISION_STYLE: Record<ExamResult["engine"]["referral"]["decision"], string> = {
  REFER: "bg-red-50 border-red-200 text-red-800",
  REPEAT_TEST: "bg-amber-50 border-amber-200 text-amber-900",
  NO_REFERRAL: "bg-emerald-50 border-emerald-200 text-emerald-800",
};

function Results({ result, plan, onRetake, onReport }: {
  result: ExamResult; plan: ExamProtocolPlan | null; onRetake: (position: string) => void;
  onReport: (format: "pdf" | "html", label: string, age: string) => Promise<string | null>;
}) {
  const [patientLabel, setPatientLabel] = useState("");
  const [age, setAge] = useState("");
  const [busy, setBusy] = useState<"pdf" | "html" | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const report = async (format: "pdf" | "html") => {
    setBusy(format);
    setReportError(await onReport(format, patientLabel, age));
    setBusy(null);
  };
  const { qc, calibration: cal, measurement, engine } = result;
  const label = (name: string) => {
    const p = plan?.positions.find((q) => q.position === name);
    return p ? positionLabel(p) : name;
  };
  const h = cal.hirschberg;
  const referral = engine.referral;
  const cells = measurement.measurements.grid.cells;
  const tiltM = measurement.measurements.tilt;
  const findings = findingsToShow(engine);
  return (
    <div className="surface-raised p-5 space-y-5">
      <div className={`rounded-md border px-4 py-3 ${DECISION_STYLE[referral.decision]}`}>
        <p className="eyebrow">{referral.urgent ? "URGENT" : "Result"}</p>
        <h3 className="display-md text-[20px] mt-1">{REFERRAL_TITLE[referral.decision]}</h3>
        <p className="text-[13px] mt-1">{referral.advice}</p>
        {referral.urgent_unconfirmed && (
          <p className="text-[13px] mt-1 font-semibold">A possible URGENT red flag was seen at low confidence.</p>
        )}
      </div>

      <div className="rounded-md border border-ink-100 p-3 space-y-2">
        <p className="eyebrow">Clinician report</p>
        <div className="flex flex-wrap gap-2">
          <input className="flex-1 min-w-[10rem] border border-ink-200 rounded-md px-3 py-2 text-[13.5px]" placeholder="Patient name or ID (optional)" value={patientLabel}
                 maxLength={120} onChange={(e) => setPatientLabel(e.target.value)} />
          <input className="w-24 border border-ink-200 rounded-md px-3 py-2 text-[13.5px]" placeholder="Age" inputMode="numeric" value={age}
                 onChange={(e) => setAge(e.target.value.replace(/[^0-9]/g, "").slice(0, 3))} />
        </div>
        <p className="text-[11.5px] text-ink-400">Printed on the report only — never stored.</p>
        <div className="flex flex-wrap gap-2">
          <button className="btn-primary" disabled={busy !== null} onClick={() => report("pdf")}>
            {busy === "pdf" ? "Preparing…" : "Open PDF report"}
          </button>
          <button className="btn-secondary" disabled={busy !== null} onClick={() => report("html")}>
            {busy === "html" ? "Preparing…" : "Open HTML report"}
          </button>
        </div>
        {reportError && <p className="text-[13px] text-red-600">{reportError}</p>}
      </div>

      <div>
        <p className="eyebrow">Deviation by gaze, PD · as you face the patient</p>
        <div className="mt-2 grid grid-cols-3 gap-1.5">
          {gridLayout(measurement.protocol).flat().map((gaze, i) => {
            if (!gaze) return <div key={i} />;
            const c = cells[gaze];
            const ok = c?.status === "OK" && c.horizontal_pd && c.vertical_pd;
            return (
              <div key={gaze} className={`rounded-md border p-2 text-[11.5px] leading-snug ${ok ? "border-ink-100" : "border-dashed border-ink-200 bg-ink-50"}`}>
                <div className="flex justify-between text-ink-400">
                  <span>{gazeName(gaze)}</span>
                  {ok && c.horizontal_pd && c.horizontal_pd.confidence !== "HIGH" && <span>{c.horizontal_pd.confidence}</span>}
                </div>
                {ok && c.horizontal_pd && c.vertical_pd ? (
                  <div className="mt-0.5 space-y-1">
                    <CellValue m={c.horizontal_pd} kind={horizontalKind(c.horizontal_pd)} />
                    <CellValue m={c.vertical_pd} kind={verticalKind(c.vertical_pd)} />
                  </div>
                ) : (
                  <button className="text-blue-600 hover:underline" onClick={() => onRetake(gaze)}>Not measured · retake</button>
                )}
              </div>
            );
          })}
        </div>
        {tiltM && (
          <p className="mt-2 text-[12.5px] text-ink-600">
            Head tilt right: {tiltM.right ? verticalLabel(tiltM.right) : "not measured"} · left:{" "}
            {tiltM.left ? verticalLabel(tiltM.left) : "not measured"}
          </p>
        )}
        {measurement.flags.length > 0 && (
          <ul className="mt-2 text-[12.5px] text-amber-800 space-y-0.5">
            {measurement.flags.filter((f) => f.code !== "duction_not_measured").map((f, i) => <li key={i}>⚠ {f.detail}</li>)}
          </ul>
        )}
      </div>

      <div>
        <p className="eyebrow">Findings</p>
        {findings.length === 0 ? (
          <p className="mt-1 text-[13px] text-ink-600">No deviation or pattern beyond measurement error.</p>
        ) : (
          <ul className="mt-2 space-y-2">
            {findings.map((f) => (
              <li key={`${f.rule_id}${f.affected_eye ?? ""}`} className="text-[13px]">
                <div className="flex gap-2 items-baseline">
                  <span className={`font-semibold ${f.urgency ? "text-red-700" : f.status === "INCONCLUSIVE" ? "text-ink-500" : "text-ink-900"}`}>
                    {f.label}{f.affected_eye ? ` (${EYE_NAME[f.affected_eye].toLowerCase()})` : ""}
                  </span>
                  <span className="text-[11px] text-ink-400">{f.status === "INCONCLUSIVE" ? "inconclusive" : f.confidence.toLowerCase()}</span>
                </div>
                <p className="text-ink-600">{f.wording}</p>
                {f.status === "INCONCLUSIVE" && f.reasons[0] && <p className="text-[12px] text-ink-400">{f.reasons[0].detail}</p>}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <p className="eyebrow">Capture quality</p>
        <h3 className="text-[15px] font-semibold text-ink-900 mt-1">
          {qc.insufficient_positions.length === 0
            ? "Every position has enough valid frames"
            : `${qc.insufficient_positions.length} position${qc.insufficient_positions.length > 1 ? "s need" : " needs"} a retake`}
        </h3>
        {qc.session_flags.length > 0 && (
          <ul className="mt-2 text-[12.5px] text-amber-800 space-y-0.5">
            {qc.session_flags.map((f) => <li key={f.code}>⚠ {f.detail}</li>)}
          </ul>
        )}
      </div>

      <table className="w-full text-[12.5px]">
        <thead className="text-ink-400 text-left"><tr><th className="font-normal">Position</th><th className="font-normal">Valid</th><th className="font-normal">Main rejections</th><th /></tr></thead>
        <tbody>
          {qc.positions.map((p) => {
            const top = Object.entries(p.reject_counts).sort((a, b) => b[1] - a[1]).slice(0, 2);
            return (
              <tr key={p.position} className="border-t border-ink-100">
                <td className="py-1.5 pr-2">{label(p.position)}</td>
                <td className={p.status === "OK" ? "text-emerald-600" : "text-red-600"}>{p.frames_valid}/{p.frames_total}</td>
                <td className="text-ink-500">{p.reason === "not_captured" ? "not captured" : top.map(([k, v]) => `${k.replace(/_/g, " ")} ×${v}`).join(", ") || "–"}</td>
                <td className="text-right">{p.status !== "OK" && <button className="text-blue-600 hover:underline" onClick={() => onRetake(p.position)}>Retake</button>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div>
        <p className="eyebrow">Calibration</p>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[13px]">
          <dt className="text-ink-500">Hirschberg ratio</dt>
          <dd className="font-mono">{fmt(h.pd_per_mm, 1)} PD/mm · {h.source === "fitted" ? "fitted for this patient" : "KB default"}</dd>
          {(["R", "L"] as ExamEye[]).map((eye) => {
            const k = cal.kappa[eye];
            return [
              <dt key={`${eye}t`} className="text-ink-500">{EYE_NAME[eye]} kappa</dt>,
              <dd key={`${eye}d`} className="font-mono">{k ? `${fmt(k.temporal_mm)} mm temporal, ${fmt(k.inferior_mm)} mm inferior` : "not measured"}</dd>,
            ];
          })}
        </dl>
        {cal.flags.length > 0 && (
          <ul className="mt-2 text-[12.5px] text-amber-800 space-y-0.5">
            {cal.flags.map((f) => <li key={`${f.code}${f.eye ?? ""}`}>⚠ {f.detail}</li>)}
          </ul>
        )}
      </div>

      <p className="text-[11.5px] text-ink-400">
        KB {result.versions.kb} · QC {result.versions.qc} · calibration {result.versions.calibration} · measurement{" "}
        {result.versions.measurement} · engine {result.versions.engine}. {result.positioning}
      </p>
    </div>
  );
}
