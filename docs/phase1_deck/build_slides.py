#!/usr/bin/env python3
"""
Generate the 13 Phase-1 deck artboards as .dc.html files.

Visual system is lifted from the BeanHealth CLR app tokens
(globals.css): ink #0f172a, muted #64748b, border #e2e8f0,
surface #f8fafc, accent #2563eb, plus the urgency ramp.
Type is IBM Plex Sans / IBM Plex Mono — technical, clinical,
and distinct from the app's UI font so the deck reads as a deck.
"""

import os

W, H = 1280, 720

HEAD = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
  <style>
    body { margin: 0; font-family: 'IBM Plex Sans', system-ui, sans-serif; }
    a { color: #2563eb; } a:hover { color: #1d4ed8; }
    * { box-sizing: border-box; }
  </style>
</helmet>
"""

FOOT = """</x-dc>
<script data-dc-script data-props='{}'>
class Component extends DCLogic {
  renderVals() { return {}; }
}
</script>
</body>
</html>
"""

# ── shared chrome ────────────────────────────────────────────────
def slide(num, kicker, title, body, tint="#ffffff"):
    """Standard slide frame: kicker rail, title, body, footer."""
    return f"""{HEAD}<div style="width: {W}px; height: {H}px; background: {tint}; display: flex; flex-direction: column; padding: 54px 64px 30px 64px; position: relative;">
  <div style="display: flex; align-items: baseline; gap: 14px; margin-bottom: 6px;">
    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; letter-spacing: 0.14em; color: #2563eb; text-transform: uppercase;">{kicker}</span>
    <span style="flex-grow: 1; height: 1px; background: #e2e8f0;"></span>
    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #94a3b8;">{num:02d} / 13</span>
  </div>
  <h1 style="font-size: 38px; line-height: 1.12; font-weight: 700; color: #0f172a; margin: 0 0 26px 0; letter-spacing: -0.02em; text-wrap: pretty;">{title}</h1>
  <div style="flex-grow: 1; display: flex; flex-direction: column; min-height: 0; justify-content: center;">
{body}
  </div>
  <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 14px; border-top: 1px solid #e2e8f0; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: #94a3b8;">
    <span>BeanHealth CLR &middot; Paediatric Strabismus Screening</span>
    <span>EyeQ Innovate Hackathon 2.0 &middot; Phase 1</span>
  </div>
</div>
{FOOT}"""


def card(inner, pad=20, bg="#ffffff", border="#e2e8f0", grow=False):
    g = "flex-grow: 1; height: 100%; " if grow else ""
    return (f'<div style="{g}background: {bg}; border: 1px solid {border}; '
            f'border-radius: 12px; padding: {pad}px;">{inner}</div>')


def label(t, color="#64748b"):
    return (f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 10.5px; '
            f'font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; '
            f'color: {color}; margin-bottom: 7px;">{t}</div>')


def body_text(t, size=15, color="#334155"):
    return (f'<p style="font-size: {size}px; line-height: 1.55; color: {color}; '
            f'margin: 0; text-wrap: pretty;">{t}</p>')


PLACEHOLDER = "#fef3c7"   # amber-100 fill for the two slides the user completes
PLACEHOLDER_BD = "#f59e0b"

slides = {}

# ── 01 · Project details ─────────────────────────────────────────
rows = [
    ("Project Name",         "BeanHealth CLR", True),
    ("Team Name",            "Team BeanHealth", False),
    ("Team Members",         "Harish Saravanan &nbsp;<span style='color:#b45309'>[ADD REMAINING MEMBERS]</span>", False),
    ("College / Institution","<span style='color:#b45309'>[ADD YOUR INSTITUTION]</span>", False),
    ("Mentor",               "<span style='color:#b45309'>[ADD MENTOR NAME]</span>", False),
    ("Theme / Domain",       "Paediatric Ophthalmology &middot; AI-Assisted Vision Screening", False),
]
r_html = ""
for k, v, big in rows:
    fs = "26px" if big else "18px"
    fw = "700" if big else "500"
    r_html += (f'<div style="display: flex; gap: 28px; align-items: baseline; padding: 13px 0; border-bottom: 1px solid #e2e8f0;">'
               f'<div style="width: 230px; flex-shrink: 0; font-family: \'IBM Plex Mono\', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #64748b;">{k}</div>'
               f'<div style="font-size: {fs}; font-weight: {fw}; color: #0f172a; letter-spacing: -0.01em;">{v}</div></div>')

slides["Main"] = f"""{HEAD}<div style="width: {W}px; height: {H}px; background: #ffffff; display: flex; flex-direction: column; padding: 60px 64px 30px 64px;">
  <div style="display: flex; align-items: center; gap: 13px; margin-bottom: 40px;">
    <div style="width: 42px; height: 42px; border-radius: 10px; background: #2563eb; display: flex; align-items: center; justify-content: center;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle></svg>
    </div>
    <div>
      <div style="font-size: 19px; font-weight: 700; color: #0f172a; letter-spacing: -0.01em;">BeanHealth</div>
      <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: #64748b; letter-spacing: 0.08em;">CLR SCREENING TOOL</div>
    </div>
  </div>
  <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; letter-spacing: 0.16em; color: #2563eb; text-transform: uppercase; margin-bottom: 10px;">Phase 1 Evaluation</div>
  <h1 style="font-size: 46px; line-height: 1.08; font-weight: 700; color: #0f172a; margin: 0 0 34px 0; letter-spacing: -0.025em; max-width: 900px;">Smartphone-based paediatric strabismus screening</h1>
  <div style="flex-grow: 1;">{r_html}</div>
  <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 14px; border-top: 1px solid #e2e8f0; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: #94a3b8;">
    <span>EyeQ Innovate Hackathon 2.0</span>
    <span>Dr. Agarwals Institute of Optometry</span>
  </div>
</div>
{FOOT}"""

# ── 02 · Clinical problem ────────────────────────────────────────
prob_items = [
    ("The clinical problem", "Strabismus &mdash; a persistent misalignment of the two visual axes. One eye fixates; the other deviates inward, outward, up or down."),
    ("Who is affected",      "Children. The condition is present from infancy or early childhood, and the child cannot report it &mdash; they do not know their vision is abnormal."),
    ("Why it matters",       "Untreated strabismus drives amblyopia: the brain suppresses the deviating eye and the visual pathway fails to develop. The treatable window narrows sharply after about age 7."),
]
cur = [
    ("Current practice", "Hirschberg corneal light reflex test and cover / cover-uncover test, performed in a clinic by an ophthalmologist or optometrist."),
    ("The unmet need",   "Both require a trained specialist and a clinic visit. Screening therefore does not reach the places children actually are &mdash; schools, anganwadis, rural PHCs and homes. Most cases are found late, after the amblyopia window has closed."),
]
left = "".join(f'<div style="margin-bottom: 17px;">{label(k)}{body_text(v)}</div>' for k, v in prob_items)
right = "".join(f'<div style="margin-bottom: 16px;">{label(k, "#b45309")}{body_text(v)}</div>' for k, v in cur)

slides["Problem"] = slide(2, "Clinical Problem", "What problem are we solving?", f"""    <div style="display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 26px; flex-grow: 1;">
      <div style="grid-column: span 3;">
        {card(f'''<div style="font-size: 54px; font-weight: 700; color: #2563eb; letter-spacing: -0.03em; line-height: 1;">1 in 30</div>
        <div style="font-size: 14px; color: #475569; margin-top: 8px; line-height: 1.45;">children are affected by strabismus</div>
        <div style="height: 1px; background: #e2e8f0; margin: 18px 0;"></div>
        <div style="font-size: 30px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em; line-height: 1;">Age 7</div>
        <div style="font-size: 14px; color: #475569; margin-top: 8px; line-height: 1.45;">age after which the amblyopia treatment window narrows sharply</div>''', bg="#f8fafc", grow=True)}
      </div>
      <div style="grid-column: span 5;">{left}</div>
      <div style="grid-column: span 4;">{right}</div>
    </div>""")

# ── 03 · Clinical learning (USER FILLS) ──────────────────────────
cl_prompts = [
    ("Key clinical concept learned",              "e.g. how the corneal light reflex localises the visual axis, and why symmetry between the two eyes &mdash; not absolute reflex position &mdash; is the diagnostic signal."),
    ("Relevant anatomy / pathology",              "e.g. cornea, pupil, visual axis, angle kappa; extraocular muscle imbalance; the suppression pathway that leads to amblyopia."),
    ("How the condition is currently assessed",   "e.g. Hirschberg reflex estimation, Krimsky, cover / cover-uncover testing, prism measurement of the deviation."),
    ("Clinical parameters relevant to our solution", "e.g. 1&nbsp;mm reflex decentration &asymp; 7&deg; of deviation; prism dioptres = 100&middot;tan&thinsp;&theta;; angle kappa as a confounder; referral thresholds by deviation size."),
]
cl = ""
for k, v in cl_prompts:
    cl += (f'<div style="background: #ffffff; border: 1px solid #fcd34d; border-radius: 10px; padding: 15px 17px; margin-bottom: 11px;">'
           f'{label(k, "#b45309")}'
           f'<div style="font-size: 13.5px; line-height: 1.5; color: #a16207; font-style: italic;">{v}</div></div>')

slides["Learning"] = slide(3, "Clinical Learning", "What did we learn from the clinical sessions?", f"""    <div style="background: {PLACEHOLDER}; border: 1px dashed {PLACEHOLDER_BD}; border-radius: 12px; padding: 15px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#b45309" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;"><path d="M12 9v4"></path><path d="M12 17h.01"></path><circle cx="12" cy="12" r="9"></circle></svg>
      <div style="font-size: 13.5px; color: #92400e; line-height: 1.45;"><strong>To be completed by the team.</strong> Replace each italic prompt below with what you actually took from the Dr. Agarwals Institute of Optometry sessions. Keep only what is relevant to this project.</div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 11px;">{cl}</div>""", tint="#fffbeb")

# ── 04 · Problem understanding (flow) ────────────────────────────
flow = [
    ("Clinical knowledge provided", "Hirschberg CLR is the standard first-line estimation test for ocular alignment; reflex decentration converts to an approximate clinical angle.", "#eff6ff", "#bfdbfe"),
    ("Understanding of the problem", "The test itself needs only a light source and an observer &mdash; what it actually needs is the trained eye that interprets it. That interpretation is the bottleneck, not the equipment.", "#eff6ff", "#bfdbfe"),
    ("The specific problem we address", "Automate the interpretation step, so a non-specialist can obtain the same objective measurement and a defensible referral decision.", "#dbeafe", "#93c5fd"),
    ("Our proposed solution", "A phone camera and its torch reproduce the reflex; computer vision measures the bilateral asymmetry and returns a triage tier with a clinical angle.", "#2563eb", "#2563eb"),
]
fh = ""
for i, (t, d, bg, bd) in enumerate(flow):
    is_last = i == len(flow) - 1
    tc = "#ffffff" if is_last else "#0f172a"
    dc = "#dbeafe" if is_last else "#475569"
    nc = "#93c5fd" if is_last else "#2563eb"
    fh += (f'<div style="background: {bg}; border: 1px solid {bd}; border-radius: 12px; padding: 17px 20px; display: flex; gap: 18px; align-items: flex-start;">'
           f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 12px; font-weight: 600; color: {nc}; padding-top: 2px;">0{i+1}</div>'
           f'<div><div style="font-size: 17px; font-weight: 600; color: {tc}; margin-bottom: 5px; letter-spacing: -0.01em;">{t}</div>'
           f'<div style="font-size: 14px; line-height: 1.5; color: {dc}; text-wrap: pretty;">{d}</div></div></div>')
    if not is_last:
        fh += ('<div style="display: flex; justify-content: center; padding: 3px 0;">'
               '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"></path><path d="m19 12-7 7-7-7"></path></svg></div>')

slides["Understanding"] = slide(4, "Problem Understanding", "How the clinical learning shaped the project", f"""    <div style="display: flex; flex-direction: column; gap: 0;">{fh}</div>""")

# ── 05 · Existing solutions ──────────────────────────────────────
ex = [
    ("Manual Hirschberg &amp; cover test", "Cover/prism cover test is the measurement gold standard; immediate; no hardware cost.", "Requires a trained specialist and a clinic visit. Cannot scale to population screening."),
    ("Photoscreeners<br><span style='font-weight:400;color:#64748b'>PlusOptix, Spot Vision</span>", "Objective, fast, validated; usable by trained non-specialists.", "Dedicated device costing several lakh rupees. Optimised for refractive error; limited alignment-angle output."),
    ("Generic smartphone vision apps", "Zero hardware cost; widely accessible.", "Predominantly acuity or colour tests. Not strabismus-specific, and generally give no clinical angle or referral tier."),
]
rows_html = ""
for n, adv, lim in ex:
    rows_html += (f'<div style="display: grid; grid-template-columns: 1.05fr 1fr 1.25fr; gap: 22px; padding: 15px 0; border-bottom: 1px solid #e2e8f0; align-items: start;">'
                  f'<div style="font-size: 15.5px; font-weight: 600; color: #0f172a; line-height: 1.35;">{n}</div>'
                  f'<div style="font-size: 13.5px; color: #15803d; line-height: 1.5;">{adv}</div>'
                  f'<div style="font-size: 13.5px; color: #b91c1c; line-height: 1.5;">{lim}</div></div>')

slides["Existing"] = slide(5, "Existing Solutions", "What already exists &mdash; and the gap", f"""    <div style="display: grid; grid-template-columns: 1.05fr 1fr 1.25fr; gap: 22px; padding-bottom: 9px; border-bottom: 1.5px solid #0f172a;">
      {label("Approach")}{label("Advantage")}{label("Limitation")}
    </div>
    {rows_html}
    <div style="margin-top: auto; padding-top: 18px;">
      {card(f'''<div style="display: flex; gap: 20px; align-items: flex-start;">
        <div style="flex-shrink: 0;">{label("Our opportunity", "#1d4ed8")}</div>
        <div style="font-size: 15.5px; line-height: 1.55; color: #1e3a8a; text-wrap: pretty;">No existing option is simultaneously <strong>zero-hardware</strong>, <strong>operable by a non-specialist</strong>, and able to return a <strong>clinical angle with a referral tier</strong>. That intersection is the gap BeanHealth CLR addresses.</div>
      </div>''', bg="#eff6ff", border="#bfdbfe")}
    </div>""")

# ── 06 · Proposed solution ───────────────────────────────────────
modules = ["Eye detection", "Pupil centre", "CLR detection", "Displacement",
           "Asymmetry (BAV)", "Classification", "Report", "Corner alignment"]
mh = ""
for i, m in enumerate(modules):
    accent = "#2563eb" if i < 7 else "#7c3aed"
    mh += (f'<div style="flex-grow: 1; background: #ffffff; border: 1px solid #e2e8f0; border-top: 2.5px solid {accent}; border-radius: 8px; padding: 10px 8px; text-align: center;">'
           f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 9.5px; color: {accent}; font-weight: 600; margin-bottom: 4px;">M{i+1}</div>'
           f'<div style="font-size: 11px; color: #334155; line-height: 1.3;">{m}</div></div>')

sol = [
    ("What we are building", "A mobile web app that automates the bilateral Hirschberg corneal light reflex test. The phone torch creates the reflex; the camera captures it; computer vision measures how far each reflex sits from its own pupil centre and compares the two eyes."),
    ("Intended user", "Parents, ASHA workers, school nurses and primary-care physicians &mdash; no ophthalmic training assumed."),
    ("Intended use", "A screening aid that decides whether a child needs to see a specialist. It is explicitly not a diagnostic device."),
]
sh = "".join(f'<div style="margin-bottom: 15px;">{label(k)}{body_text(v, 14)}</div>' for k, v in sol)

slides["Solution"] = slide(6, "Proposed Solution", "What are we building?", f"""    <div style="display: grid; grid-template-columns: 1.15fr 1fr; gap: 30px; flex-grow: 1;">
      <div>{sh}</div>
      <div>
        {label("What makes it different")}
        <div style="display: flex; flex-direction: column; gap: 9px;">
          {card('<div style="font-size: 13.5px; line-height: 1.5; color: #334155;"><strong style="color:#0f172a">Bilateral asymmetry, not absolute displacement.</strong> Comparing the two eyes largely cancels the symmetric component of angle kappa, the anatomical offset that makes normal eyes look deviated.</div>', pad=14, bg="#f8fafc")}
          {card('<div style="font-size: 13.5px; line-height: 1.5; color: #334155;"><strong style="color:#0f172a">Five states, not four.</strong> A variance gate returns INCONCLUSIVE rather than a number it cannot stand behind.</div>', pad=14, bg="#f8fafc")}
          {card('<div style="font-size: 13.5px; line-height: 1.5; color: #334155;"><strong style="color:#0f172a">Two independent methods.</strong> Hirschberg where a torch reflex exists, plus a torch-free pupil-vs-corner alignment check that works on ordinary photos.</div>', pad=14, bg="#f8fafc")}
        </div>
      </div>
    </div>
    <div style="margin-top: 18px;">
      {label("Pipeline &mdash; phone photo to clinical triage")}
      <div style="display: flex; gap: 6px;">{mh}</div>
    </div>""")

# ── 07 · Architecture ────────────────────────────────────────────
comps = [
    ("Component 1", "Core screening engine &amp; clinical pipeline hardening", "Phase 1", True),
    ("Component 2", "Per-phone calibration &amp; ground-truth dataset", "Phase 2", False),
    ("Component 3", "Screener app, report &amp; referral pathway", "Phase 3", False),
    ("Component 4", "Validation, field testing &amp; MVP demonstration", "Phase 4", False),
]
ch = ""
for i, (c, d, ph, active) in enumerate(comps):
    bg = "#2563eb" if active else "#ffffff"
    bd = "#2563eb" if active else "#e2e8f0"
    tc = "#ffffff" if active else "#0f172a"
    dc = "#dbeafe" if active else "#64748b"
    pc = "#bfdbfe" if active else "#94a3b8"
    ch += (f'<div style="flex-grow: 1; flex-basis: 0; background: {bg}; border: 1.5px solid {bd}; border-radius: 12px; padding: 18px 16px; display: flex; flex-direction: column;">'
           f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; color: {pc}; text-transform: uppercase; margin-bottom: 8px;">{ph}</div>'
           f'<div style="font-size: 16px; font-weight: 700; color: {tc}; margin-bottom: 7px; letter-spacing: -0.01em;">{c}</div>'
           f'<div style="font-size: 12.5px; line-height: 1.45; color: {dc};">{d}</div></div>')
    if i < 3:
        ch += ('<div style="display: flex; align-items: center; padding: 0 3px;">'
               '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path><path d="m12 5 7 7-7 7"></path></svg></div>')

slides["Architecture"] = slide(7, "Product Architecture", "The complete development pathway", f"""    <div style="display: flex; align-items: stretch; margin-bottom: 8px;">{ch}
      <div style="display: flex; align-items: center; padding: 0 3px;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path><path d="m12 5 7 7-7 7"></path></svg>
      </div>
      <div style="flex-shrink: 0; width: 150px; background: #0f172a; border-radius: 12px; padding: 18px 16px; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">Grand Finale</div>
        <div style="font-size: 16px; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;">Final MVP</div>
      </div>
    </div>
    <div style="display: flex; justify-content: flex-start; margin-top: 6px;">
      <div style="width: 232px; text-align: center; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; font-weight: 600; letter-spacing: 0.1em; color: #2563eb;">&#9650; PHASE 1 SCOPE</div>
    </div>
    <div style="margin-top: auto;">
      {card(f'''{label("What Component 1 contributes to the final product", "#1d4ed8")}
      <div style="font-size: 15px; line-height: 1.55; color: #1e3a8a; text-wrap: pretty;">Component 1 is the measurement engine every later component depends on. Calibration (C2) tunes its thresholds, the app (C3) is its user interface, and validation (C4) measures its output against clinician ground truth. If the engine is not deterministic and confidence-gated, nothing built on top of it can be trusted &mdash; which is why it is Phase 1.</div>''', bg="#eff6ff", border="#bfdbfe")}
    </div>""")

# ── 08 · Phase 0 -> Phase 1 -> Phase 2 ───────────────────────────
# Anchored to the five innovations claimed in the Phase 0 pitch deck, so a
# judge holding that deck can check each promise against what now runs.
phases = [
    ("Alignment scoring",
     "Innovation 01 &mdash; bilateral asymmetry to cancel kappa",
     "Built. BAV cancels the symmetric kappa component.", "done"),
    ("Confidence gating",
     "Innovation 02 &mdash; inter-frame variance gate",
     "Built. 5th state: INCONCLUSIVE rather than a false referral.", "done"),
    ("Capture quality",
     "Innovation 04 &mdash; active quality gating",
     "Built. Live 5-point meter + paediatric attention-getter.", "done"),
    ("Measurement precision",
     "Not addressed",
     "Native-resolution face crop, sub-pixel centroids, head-roll de-rotation.", "new"),
    ("Screening methods",
     "Hirschberg only (torch required)",
     "Module 8 added &mdash; torch-free pupil-vs-corner alignment.", "new"),
    ("Throughput",
     "Single patient",
     "Batch pre-screen &mdash; a whole class in one pass.", "new"),
    ("Device handling",
     "Innovation 03 &mdash; per-phone calibration database",
     "Session-level auto-calibration only.", "partial"),
    ("Referral loop",
     "Innovation 05 &mdash; booking, SMS/IVR, adherence",
     "Annotated PDF report; loop not yet closed.", "partial"),
    ("Clinical evidence",
     "Target &ge;85% sensitivity / &ge;85% specificity vs PACT",
     "No clinical dataset yet &mdash; not validated.", "todo"),
]
NEXT = {
    "Alignment scoring":     "Validate BAV against PACT",
    "Confidence gating":     "Tune thresholds on real data",
    "Capture quality":       "Cover-test capture protocol",
    "Measurement precision": "On-device movement analysis",
    "Screening methods":     "Cover-test refixation detection",
    "Throughput":            "School / camp deployment",
    "Device handling":       "Persistent per-device profiles",
    "Referral loop":         "Booking + SMS/IVR + adherence",
    "Clinical evidence":     "Labelled dataset &rarr; sens/spec",
}
MARK = {
    "done":    ("#16a34a", "&#10003;"),
    "new":     ("#2563eb", "&#10003;"),
    "partial": ("#b45309", "&#8226;"),
    "todo":    ("#94a3b8", "&#8226;"),
}
ph = ""
for cap, p0, p1, state in phases:
    col, glyph = MARK[state]
    ph += (f'<div style="display: grid; grid-template-columns: 0.62fr 1fr 1.22fr 0.86fr; gap: 14px; '
           f'padding: 7.5px 0; border-bottom: 1px solid #f1f5f9; align-items: baseline;">'
           f'<div style="font-size: 12px; font-weight: 600; color: #0f172a;">{cap}</div>'
           f'<div style="font-size: 11.5px; color: #94a3b8; line-height: 1.4;">{p0}</div>'
           f'<div style="font-size: 11.5px; color: #0f172a; line-height: 1.4;">'
           f'<span style="color: {col}; font-weight: 700;">{glyph}</span> {p1}</div>'
           f'<div style="font-size: 11.5px; color: #6d28d9; line-height: 1.4;">{NEXT[cap]}</div></div>')

slides["Component1"] = slide(8, "Component 1 Development",
    "Phase 0 proposed &rarr; Phase 1 delivered &rarr; Phase 2 next", f"""    <div style="display: grid; grid-template-columns: 0.62fr 1fr 1.22fr 0.86fr; gap: 14px; padding-bottom: 7px; border-bottom: 1.5px solid #0f172a;">
      {label("Capability")}{label("Phase 0 &mdash; proposed")}{label("Phase 1 &mdash; delivered", "#2563eb")}{label("Phase 2 &mdash; next", "#6d28d9")}
    </div>
    {ph}
    <div style="margin-top: 14px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;">
      {card('<div style="font-size: 11px; color: #64748b; line-height: 1.45;"><strong style="color:#0f172a">3 of 5</strong> Phase 0 innovations now running in code, plus three capabilities that were never proposed.</div>', pad=12, bg="#f8fafc")}
      {card('<div style="font-size: 11px; color: #64748b; line-height: 1.45;">Angular error per pixel cut <strong style="color:#0f172a">2.96&deg; &rarr; 2.12&deg;</strong>; head roll no longer leaks a horizontal deviation into the vertical axis.</div>', pad=12, bg="#f8fafc")}
      {card('<div style="font-size: 11px; color: #5b21b6; line-height: 1.45;">Phase 2 moves from <strong>static position</strong> to <strong>movement</strong> &mdash; the construct PACT actually measures.</div>', pad=12, bg="#f5f3ff", border="#ddd6fe")}
    </div>""")

# ── 09 · Demonstration ───────────────────────────────────────────
slides["Demonstration"] = slide(9, "Component 1 Demonstration", "Proof of concept &mdash; the working system", f"""    <div style="display: grid; grid-template-columns: 1.32fr 1fr; gap: 24px; flex-grow: 1; min-height: 0;">
      <div style="display: flex; flex-direction: column; min-height: 0;">
        {label("Measurement stage &mdash; live pipeline output")}
        <div style="border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; background: #0f172a;">
          <img src="pipeline_module5_vector.jpg" alt="Per-eye pupil, corneal reflex and iris boundary with displacement vector" style="width: 100%; display: block;">
        </div>
        <div style="display: flex; gap: 16px; margin-top: 11px; font-size: 11.5px; color: #475569;">
          <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 9px; height: 9px; border-radius: 50%; background: #3b82f6; display: inline-block;"></span>Pupil centre</span>
          <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 9px; height: 9px; border-radius: 50%; background: #f97316; display: inline-block;"></span>Corneal light reflex</span>
          <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 9px; height: 9px; border-radius: 50%; border: 1.5px solid #22c55e; display: inline-block;"></span>Iris boundary</span>
        </div>
        <div style="margin-top: 12px;">
          {card('''<div style="font-size: 13px; line-height: 1.5; color: #334155;">Each eye is measured independently: reflex displacement from the pupil centre, normalised by iris radius so the result is scale-invariant. The two normalised vectors are then subtracted &mdash; that difference, not either eye alone, is the screening signal.</div>''', pad=14, bg="#f8fafc")}
        </div>
      </div>
      <div style="display: flex; flex-direction: column; min-height: 0;">
        {label("Screener application &mdash; deployed")}
        <div style="border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; flex-grow: 1; min-height: 0; display: flex; align-items: flex-start; justify-content: center; background: #f8fafc;">
          <img src="ui_landing.jpg" alt="BeanHealth CLR screener application" style="max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; display: block;">
        </div>
      </div>
    </div>""")

# ── 10 · Mentor feedback (USER FILLS) ────────────────────────────
fb = [("Clinical feedback", "#2563eb"), ("Technical feedback", "#7c3aed"), ("Product feedback", "#0d9488")]
fh2 = ""
for k, c in fb:
    fh2 += (f'<div style="display: grid; grid-template-columns: 0.62fr 1fr 1fr; gap: 18px; padding: 17px 0; border-bottom: 1px solid #fcd34d; align-items: start;">'
            f'<div style="font-size: 15px; font-weight: 600; color: {c};">{k}</div>'
            f'<div style="font-size: 13px; color: #a16207; font-style: italic; line-height: 1.5;">[What the mentor said]</div>'
            f'<div style="font-size: 13px; color: #a16207; font-style: italic; line-height: 1.5;">[What we changed, or will change]</div></div>')

slides["Feedback"] = slide(10, "Mentor Feedback", "What feedback did we receive?", f"""    <div style="background: {PLACEHOLDER}; border: 1px dashed {PLACEHOLDER_BD}; border-radius: 12px; padding: 15px 18px; margin-bottom: 18px; display: flex; align-items: center; gap: 12px;">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#b45309" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;"><path d="M12 9v4"></path><path d="M12 17h.01"></path><circle cx="12" cy="12" r="9"></circle></svg>
      <div style="font-size: 13.5px; color: #92400e; line-height: 1.45;"><strong>To be completed by the team.</strong> Record the actual feedback received from your mentor and what it changed in the product. Judges may verify this with the mentor &mdash; keep it factual.</div>
    </div>
    <div style="display: grid; grid-template-columns: 0.62fr 1fr 1fr; gap: 18px; padding-bottom: 9px; border-bottom: 1.5px solid #b45309;">
      {label("Source", "#b45309")}{label("Feedback", "#b45309")}{label("What we changed / will change", "#b45309")}
    </div>
    {fh2}
    <div style="margin-top: auto; padding-top: 14px; font-size: 13px; color: #92400e; line-height: 1.5;">This slide demonstrates that mentoring is actively influencing product development.</div>""", tint="#fffbeb")

# ── 11 · Current status ──────────────────────────────────────────
status = [
    ("Component 1", "Core screening engine", "Phase 1", "complete"),
    ("Component 2", "Calibration &amp; ground-truth dataset", "Phase 2", "partial"),
    ("Component 3", "Screener app &amp; referral pathway", "Phase 3", "started"),
    ("Component 4", "Validation &amp; MVP demonstration", "Phase 4", "pending"),
]
STYLES = {
    "complete": ("#16a34a", "#f0fdf4", "#bbf7d0", "Delivered"),
    "started":  ("#2563eb", "#eff6ff", "#bfdbfe", "Ahead of schedule &mdash; substantially delivered"),
    "partial":  ("#7c3aed", "#f5f3ff", "#ddd6fe", "In progress &mdash; session calibration in place"),
    "pending":  ("#94a3b8", "#f8fafc", "#e2e8f0", "Planned"),
}
sth = ""
for c, d, ph, st in status:
    col, bg, bd, lbl = STYLES[st]
    sth += (f'<div style="display: grid; grid-template-columns: 0.5fr 1.15fr 0.4fr 1fr; gap: 18px; padding: 15px 0; border-bottom: 1px solid #e2e8f0; align-items: center;">'
            f'<div style="font-size: 15px; font-weight: 600; color: #0f172a;">{c}</div>'
            f'<div style="font-size: 13.5px; color: #475569;">{d}</div>'
            f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 12px; color: #64748b;">{ph}</div>'
            f'<div><span style="display: inline-block; background: {bg}; border: 1px solid {bd}; color: {col}; font-size: 12px; font-weight: 600; padding: 4px 11px; border-radius: 20px;">{lbl}</span></div></div>')

slides["Status"] = slide(11, "Current Project Status", "Where are we now?", f"""    <div style="display: grid; grid-template-columns: 0.5fr 1.15fr 0.4fr 1fr; gap: 18px; padding-bottom: 9px; border-bottom: 1.5px solid #0f172a;">
      {label("Component")}{label("Scope")}{label("Due")}{label("Status")}
    </div>
    {sth}
    <div style="margin-top: auto; padding-top: 18px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px;">
      {card('<div style="font-size: 30px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">8</div><div style="font-size: 12.5px; color: #64748b; margin-top: 4px;">pipeline modules, up from 7</div>', pad=15, bg="#f8fafc")}
      {card('<div style="font-size: 30px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">5</div><div style="font-size: 12.5px; color: #64748b; margin-top: 4px;">triage states, up from 4</div>', pad=15, bg="#f8fafc")}
      {card('<div style="font-size: 30px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">2</div><div style="font-size: 12.5px; color: #64748b; margin-top: 4px;">independent screening methods, up from 1</div>', pad=15, bg="#f8fafc")}
    </div>""")

# ── 12 · Phase 2 plan ────────────────────────────────────────────
p2 = [
    ("What Component 2 will do", "Establish cross-device consistency and build the clinical evidence base: a per-phone calibration profile (torch-to-lens offset, sensor and exposure behaviour) applied to every capture, and a structured ground-truth dataset pairing smartphone captures with clinician reference assessment."),
    ("How it connects to Component 1", "Component 1 currently calibrates within a session. Component 2 turns that into a persistent per-device profile and supplies the labelled data needed to tune Component 1's thresholds against real clinical outcomes rather than assumption."),
]
p2h = "".join(f'<div style="margin-bottom: 17px;">{label(k)}{body_text(v, 14)}</div>' for k, v in p2)
reqs = [
    ("Technical requirements", "Device fingerprinting and a calibration profile store; a capture-and-label workflow; anonymisation and consent handling in the data path."),
    ("Clinical knowledge required", "Clinician reference assessment (Hirschberg / cover test) for each captured session, and agreement on referral thresholds by deviation size."),
    ("Expected outcome", "A working per-phone calibration system covering the common device fleet, plus an initial labelled smartphone-versus-clinician dataset &mdash; the validation set Component 4 measures against."),
]
rq = ""
for k, v in reqs:
    inner = label(k) + '<div style="font-size: 13px; line-height: 1.5; color: #334155;">' + v + '</div>'
    rq += '<div style="margin-bottom: 10px;">' + card(inner, pad=14, bg="#f8fafc") + '</div>'

slides["Phase2"] = slide(12, "Phase 2 Plan", "Component 2 &mdash; calibration &amp; ground truth", f"""    <div style="display: grid; grid-template-columns: 1.1fr 1fr; gap: 30px; flex-grow: 1;">
      <div>{p2h}</div>
      <div>{rq}</div>
    </div>""")

# ── 13 · Milestone ───────────────────────────────────────────────
miles = [("Clinical concept understood", True), ("Clinical problem clearly defined", True),
         ("Proposed solution finalised", True), ("Component 1 developed &amp; deployed", True),
         ("Mentor feedback incorporated", False)]
TICK = '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;"><path d="M20 6 9 17l-5-5"></path></svg>'
OPEN = '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#b45309" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;"><circle cx="12" cy="12" r="9"></circle></svg>'
mhx = ""
for m, done in miles:
    bd = "#e2e8f0" if done else "#fcd34d"
    bg = "#ffffff" if done else "#fffbeb"
    tc = "#0f172a" if done else "#92400e"
    mhx += (f'<div style="display: flex; align-items: center; gap: 13px; padding: 13px 18px; background: {bg}; border: 1px solid {bd}; border-radius: 10px;">'
            + (TICK if done else OPEN)
            + f'<span style="font-size: 15.5px; color: {tc}; font-weight: 500;">{m}</span></div>')

slides["Milestone"] = slide(13, "Phase 1 Milestone", "Phase 1 output", f"""    <div style="display: grid; grid-template-columns: 1.15fr 1fr; gap: 34px; flex-grow: 1; align-items: start;">
      <div style="display: flex; flex-direction: column; gap: 9px;">{mhx}</div>
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"></path><path d="m19 12-7 7-7-7"></path></svg>
        <div style="margin-top: 18px; background: #0f172a; border-radius: 14px; padding: 32px 40px; text-align: center; width: 100%;">
          <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.16em; color: #64748b; text-transform: uppercase; margin-bottom: 10px;">Status</div>
          <div style="font-size: 30px; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; line-height: 1.2;">Ready for<br>Phase 2</div>
        </div>
        <div style="margin-top: 16px; font-size: 13px; color: #64748b; text-align: center; line-height: 1.5;">Phase 1 = clinical understanding + problem definition + Component 1 development</div>
      </div>
    </div>""", tint="#f8fafc")

# ── write ────────────────────────────────────────────────────────
here = os.path.dirname(os.path.abspath(__file__))
for name, src in slides.items():
    with open(os.path.join(here, f"{name}.dc.html"), "w", encoding="utf-8") as f:
        f.write(src)
print("wrote:", ", ".join(f"{n}.dc.html" for n in slides))
