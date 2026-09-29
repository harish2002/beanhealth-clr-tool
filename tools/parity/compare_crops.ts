/**
 * Browser/server parity — step 2 of 2: run the browser port on the dumped crops.
 * See tools/parity/dump_crops.py for step 1.
 *
 *     node tools/parity/compare_crops.ts /tmp/parity
 *
 * Exits non-zero if any pupil or reflex differs by more than 0.001 px, or any
 * outcome (found / no reflex / no flash) differs.
 */
import { readFileSync } from "node:fs";
import { darkPupilCentre, detectReflex, rgbaToGray } from "../../frontend/lib/eyeVision.ts";

const dir = process.argv[2] ?? "/tmp/parity";
const TOL = 0.001;
interface Case {
  name: string; photo: string; eye: string; variant: string; w: number; h: number;
  seed: [number, number]; iris_r: number; server_pupil: [number, number] | null;
  pupil_for_reflex: [number, number];
  server_reflex: { status: string; x?: number; y?: number };
}
const cases: Case[] = JSON.parse(readFileSync(`${dir}/cases.json`, "utf8"));
const problems: string[] = [];
let pupils = 0, reflexes = 0, outcomes = 0, worst = 0;

for (const c of cases) {
  const gray = rgbaToGray(new Uint8ClampedArray(readFileSync(`${dir}/${c.name}.rgba`)));
  const where = `${c.photo} ${c.eye} ${c.variant}`;

  if (c.variant === "real") {
    const p = darkPupilCentre(gray, c.w, c.h, { x: c.seed[0], y: c.seed[1] }, c.iris_r);
    if (!p !== !c.server_pupil) problems.push(`${where}: pupil found by only one side`);
    else if (p && c.server_pupil) {
      const e = Math.hypot(p.x - c.server_pupil[0], p.y - c.server_pupil[1]);
      pupils++; worst = Math.max(worst, e);
      if (e > TOL) problems.push(`${where}: pupil differs by ${e.toFixed(4)} px`);
    }
  }

  const r = detectReflex(gray, c.w, c.h, { x: c.pupil_for_reflex[0], y: c.pupil_for_reflex[1] }, c.iris_r);
  outcomes++;
  if (r.status !== c.server_reflex.status) problems.push(`${where}: reflex ${r.status} vs server ${c.server_reflex.status}`);
  else if (r.status === "ok") {
    const e = Math.hypot(r.pos.x - (c.server_reflex.x as number), r.pos.y - (c.server_reflex.y as number));
    reflexes++; worst = Math.max(worst, e);
    if (e > TOL) problems.push(`${where}: reflex differs by ${e.toFixed(4)} px`);
  }
}

console.log(`${cases.length} crops · ${pupils} pupils and ${reflexes} reflexes compared · ${outcomes} outcomes checked`);
console.log(`largest position difference: ${worst.toExponential(2)} px`);
if (problems.length) {
  console.log(`\n${problems.length} DIFFERENCES:\n  ` + problems.join("\n  "));
  process.exit(1);
}
console.log("browser port matches the server");
