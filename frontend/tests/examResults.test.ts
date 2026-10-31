/**
 * Unit tests for lib/examResults.ts — how the gaze-exam result is laid out.
 *
 *   npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { findingsToShow, gridLayout, horizontalLabel, verticalLabel } from "../lib/examResults.ts";
import type { ExamEngineResult, ExamFinding } from "../lib/types.ts";

test("the chart faces the patient: their right gaze on the examiner's left", () => {
  const nine = gridLayout("nine_gaze");
  assert.deepEqual(nine[1], ["right", "primary", "left"]);
  assert.deepEqual(nine[0][0], "up_right");
  const five = gridLayout("five_gaze");
  assert.deepEqual(five[0], [null, "up", null]);
  assert.equal(five.flat().filter(Boolean).length, 5);
});

test("labels follow the sign convention and the error", () => {
  const m = (value: number, error = 1) => ({ value, error, confidence: "HIGH" as const });
  assert.equal(horizontalLabel(m(14.24, 1.1)), "14.2 ± 1.1 PD eso");
  assert.equal(horizontalLabel(m(-9)), "9.0 ± 1.0 PD exo");
  assert.equal(horizontalLabel(m(1.5)), "1.5 ± 1.0 PD ortho");
  assert.equal(verticalLabel(m(-6)), "6.0 ± 1.0 PD L hyper");
  assert.equal(verticalLabel(m(4)), "4.0 ± 1.0 PD R hyper");
  assert.equal(verticalLabel(m(-0.5)), "0.5 ± 1.0 PD no vertical");
});

test("urgent first, then abnormal, then inconclusive; normal findings hidden", () => {
  const f = (rule_id: string, status: ExamFinding["status"], abnormal: boolean, urgency: ExamFinding["urgency"] = null): ExamFinding => ({
    rule_id, label: rule_id, kind: "pattern", affected_eye: null, status, abnormal, urgency, confidence: "HIGH",
    margin_over_threshold: null, wording: "", reasons: [], muscle: null,
  });
  const engine: ExamEngineResult = {
    findings: [f("comitance", "REPORTED", false), f("a_pattern", "INCONCLUSIVE", true), f("eso", "REPORTED", true),
               f("cn3", "REPORTED", true, "URGENT")],
    not_evaluated: [], positioning: "", limitations: [], engine_version: "0.1.0",
    referral: { decision: "REFER", urgent: true, urgent_unconfirmed: false, because: [], advice: "" },
  };
  assert.deepEqual(findingsToShow(engine).map((x) => x.rule_id), ["cn3", "eso", "a_pattern"]);
});
