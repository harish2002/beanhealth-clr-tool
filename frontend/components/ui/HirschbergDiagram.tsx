/**
 * HirschbergDiagram — the measurement, drawn.
 *
 * Two eyes under torch light. In the left eye the corneal light reflex sits on
 * the pupil centre; in the right it is displaced nasally. The offset between
 * the two is the whole signal this tool measures — 1 mm ≈ 7° of deviation.
 *
 * Purely decorative/explanatory: no data binding, no interactivity.
 */

function Eye({
  cx,
  cy,
  /** CLR offset from pupil centre, in px */
  clrDx = 0,
  label,
  labelSide,
}: {
  cx: number;
  cy: number;
  clrDx?: number;
  label: string;
  labelSide: "left" | "right";
}) {
  const w = 78;
  const h = 30;
  const irisR  = 30;
  const pupilR = 13;
  const clrR   = 3.8;

  const lid = `M ${cx - w},${cy}
               C ${cx - w * 0.5},${cy - h * 1.7} ${cx + w * 0.5},${cy - h * 1.7} ${cx + w},${cy}
               C ${cx + w * 0.5},${cy + h * 1.7} ${cx - w * 0.5},${cy + h * 1.7} ${cx - w},${cy} Z`;

  const clipId = `eye-clip-${labelSide}`;

  return (
    <g>
      <defs>
        <clipPath id={clipId}>
          <path d={lid} />
        </clipPath>
      </defs>

      {/* Sclera */}
      <path d={lid} fill="#FBFCFE" />

      <g clipPath={`url(#${clipId})`}>
        {/* Iris */}
        <circle cx={cx} cy={cy} r={irisR} fill="#E8ECF2" />
        <circle cx={cx} cy={cy} r={irisR} fill="none" stroke="#C6CEDA" strokeWidth="1" />
        {/* Radial iris striations — subtle texture */}
        {Array.from({ length: 24 }).map((_, i) => {
          const a = (i / 24) * Math.PI * 2;
          return (
            <line
              key={i}
              x1={cx + Math.cos(a) * (pupilR + 2)}
              y1={cy + Math.sin(a) * (pupilR + 2)}
              x2={cx + Math.cos(a) * (irisR - 1)}
              y2={cy + Math.sin(a) * (irisR - 1)}
              stroke="#CFD6E1"
              strokeWidth="0.75"
            />
          );
        })}
        {/* Pupil */}
        <circle cx={cx} cy={cy} r={pupilR} fill="#0A1019" />
      </g>

      {/* Lid outline */}
      <path d={lid} fill="none" stroke="#0A1019" strokeWidth="1.75" strokeLinejoin="round" />

      {/* Pupil-centre crosshair */}
      <g stroke="#1B47C4" strokeWidth="1.1" opacity="0.9">
        <line x1={cx - 8} y1={cy} x2={cx + 8} y2={cy} />
        <line x1={cx} y1={cy - 8} x2={cx} y2={cy + 8} />
      </g>

      {/* Displacement vector */}
      {clrDx !== 0 && (
        <line
          x1={cx}
          y1={cy}
          x2={cx + clrDx}
          y2={cy}
          stroke="#0E7C7B"
          strokeWidth="1.75"
          strokeDasharray="3 2.5"
        />
      )}

      {/* Corneal light reflex */}
      <circle cx={cx + clrDx} cy={cy} r={clrR + 3} fill="#FDE68A" opacity="0.45" />
      <circle cx={cx + clrDx} cy={cy} r={clrR} fill="#FBBF24" />
      <circle cx={cx + clrDx - 1} cy={cy - 1} r={1.3} fill="#FFFDF2" />

      {/* Label */}
      <text
        x={cx}
        y={cy + 80}
        textAnchor="middle"
        fontSize="10.5"
        fontWeight="600"
        letterSpacing="1.6"
        fill="#8A94A3"
        fontFamily="var(--font-mono), monospace"
      >
        {label}
      </text>
    </g>
  );
}

export default function HirschbergDiagram({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 420 260"
      className={className}
      role="img"
      aria-label="Diagram: the torch reflex sits centred in the left pupil but displaced in the right, and that difference is the measured deviation."
    >
      {/* Torch source */}
      <g>
        <circle cx="210" cy="34" r="17" fill="#FEF3C7" />
        <circle cx="210" cy="34" r="8" fill="#FBBF24" />
        <circle cx="210" cy="34" r="3" fill="#FFFDF2" />
        {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
          const a = (deg * Math.PI) / 180;
          return (
            <line
              key={deg}
              x1={210 + Math.cos(a) * 21}
              y1={34 + Math.sin(a) * 21}
              x2={210 + Math.cos(a) * 27}
              y2={34 + Math.sin(a) * 27}
              stroke="#FBBF24"
              strokeWidth="1.75"
              strokeLinecap="round"
              opacity="0.7"
            />
          );
        })}
        <text
          x="210" y="70" textAnchor="middle"
          fontSize="9.5" fontWeight="600" letterSpacing="1.6" fill="#B45309"
          fontFamily="var(--font-mono), monospace"
        >
          TORCH
        </text>
      </g>

      {/* Light paths to each cornea */}
      <g stroke="#E8D9A8" strokeWidth="1" strokeDasharray="2 3">
        <line x1="196" y1="46" x2="115" y2="118" />
        <line x1="224" y1="46" x2="305" y2="118" />
      </g>

      <Eye cx={110} cy={140} clrDx={0}  label="ALIGNED"  labelSide="left" />
      <Eye cx={310} cy={140} clrDx={13} label="DEVIATED" labelSide="right" />

      {/* Measurement bracket under the deviated eye: pupil centre → reflex */}
      <g stroke="#0E7C7B" fill="none">
        <line x1="310" y1="182" x2="310" y2="194" strokeWidth="1" />
        <line x1="323" y1="182" x2="323" y2="194" strokeWidth="1" />
        <line x1="310" y1="192" x2="323" y2="192" strokeWidth="1.25" />
      </g>
      <text
        x="333" y="196" textAnchor="start"
        fontSize="10.5" fontWeight="600" fill="#0E7C7B"
        fontFamily="var(--font-mono), monospace"
      >
        1 mm ≈ 7°
      </text>
    </svg>
  );
}
