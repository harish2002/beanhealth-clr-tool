/**
 * BeanHealth CLR — gaze-exam results: pure display helpers
 * ========================================================
 *
 * How the /exam/evaluate result is laid out for the operator. Signs follow
 * the backend: horizontal + = eso, − = exo; vertical + = right hyper,
 * − = left hyper. Right / left are always the patient's.
 */

import type { ExamEngineResult, ExamFinding, ExamGaze, ExamMeasured, ExamProtocol } from "./types";

/**
 * The 9-gaze chart as the examiner faces the patient: the patient's right
 * gaze is on the examiner's LEFT. null = no position in this protocol.
 */
export function gridLayout(protocol: ExamProtocol): (ExamGaze | null)[][] {
  const nine: ExamGaze[][] = [
    ["up_right", "up", "up_left"],
    ["right", "primary", "left"],
    ["down_right", "down", "down_left"],
  ];
  if (protocol === "nine_gaze") return nine;
  const five = new Set<ExamGaze>(["primary", "right", "left", "up", "down"]);
  return nine.map((row) => row.map((g) => (five.has(g) ? g : null)));
}

const GAZE_NAME: Record<ExamGaze, string> = {
  primary: "Primary", right: "Right", left: "Left", up: "Up", down: "Down",
  up_right: "Up right", up_left: "Up left", down_right: "Down right", down_left: "Down left",
};

export function gazeName(g: ExamGaze): string {
  return GAZE_NAME[g];
}

/** eso / exo; "ortho" when the value is inside k × error. */
export function horizontalKind(m: ExamMeasured, k = 2): string {
  if (Math.abs(m.value) <= k * m.error) return "ortho";
  return m.value > 0 ? "eso" : "exo";
}

/** R hyper / L hyper; "no vertical" when the value is inside k × error. */
export function verticalKind(m: ExamMeasured, k = 2): string {
  if (Math.abs(m.value) <= k * m.error) return "no vertical";
  return m.value > 0 ? "R hyper" : "L hyper";
}

/** Size and error without the sign (the kind carries the direction): ["14.2", "1.1"]. */
export function sizeParts(m: ExamMeasured): [string, string] {
  return [Math.abs(m.value).toFixed(1), m.error.toFixed(1)];
}

/** "14.2 ± 1.1 PD eso" */
export function horizontalLabel(m: ExamMeasured, k = 2): string {
  const [v, e] = sizeParts(m);
  return `${v} ± ${e} PD ${horizontalKind(m, k)}`;
}

/** "6.0 ± 1.0 PD L hyper" */
export function verticalLabel(m: ExamMeasured, k = 2): string {
  const [v, e] = sizeParts(m);
  return `${v} ± ${e} PD ${verticalKind(m, k)}`;
}

/** Findings worth showing, most important first: urgent, abnormal, then inconclusive. */
export function findingsToShow(engine: ExamEngineResult): ExamFinding[] {
  const rank = (f: ExamFinding) =>
    f.urgency === "URGENT" ? 0 : f.status === "REPORTED" && f.abnormal ? 1 : f.status === "INCONCLUSIVE" ? 2 : 3;
  return engine.findings.filter((f) => rank(f) < 3).sort((a, b) => rank(a) - rank(b));
}

export const REFERRAL_TITLE: Record<ExamEngineResult["referral"]["decision"], string> = {
  REFER: "Refer to ophthalmology",
  REPEAT_TEST: "Repeat the examination",
  NO_REFERRAL: "No deviation beyond measurement error",
};
