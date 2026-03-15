import type { SuccessResponse } from '@/lib/types';

interface ProcessingStepsProps {
  intermediateImages: NonNullable<SuccessResponse['intermediate_images']>;
}

const STEPS = [
  {
    key:         'module1_crops'  as const,
    step:        '01',
    title:       'Eye Detection',
    description: 'MediaPipe Face Mesh locates both eyes and extracts tight crops. Iris landmarks (5 pts per eye) are stored for Module 3.',
    tag:         'Module 1',
    tagColour:   'bg-slate-200 text-slate-600',
  },
  {
    key:         'module2_clahe'  as const,
    step:        '02',
    title:       'Grayscale + CLAHE',
    description: 'Crops converted to grayscale then enhanced with CLAHE (Contrast Limited Adaptive Histogram Equalisation) — this boosts the bright CLR spot against the iris so the percentile threshold can isolate it.',
    tag:         'Module 2 pre-step',
    tagColour:   'bg-slate-200 text-slate-600',
  },
  {
    key:         'module3_pupil'  as const,
    step:        '03',
    title:       'Pupil Localisation',
    description: 'Two independent methods — MediaPipe iris landmark mean (blue dot) and Hough Circle Transform — are cross-validated. Agreement < 5 px → HIGH confidence; 5–15 px → MEDIUM; > 15 px → LOW.',
    tag:         'Module 2',
    tagColour:   'bg-blue-100 text-blue-700',
  },
  {
    key:         'module4_clr'    as const,
    step:        '04',
    title:       'CLR Detection',
    description: 'Top 3% brightest pixels thresholded. Connected blobs filtered by 3 rules: ① location (central 80%), ② area (0.5–15% of iris area), ③ circularity > 0.5. The largest passing blob is the corneal light reflex (amber dot).',
    tag:         'Module 3',
    tagColour:   'bg-amber-100 text-amber-700',
  },
  {
    key:         'module5_vector' as const,
    step:        '05',
    title:       'Displacement Vector',
    description: 'Vector drawn from pupil centre → CLR. Magnitude normalised by iris radius (shown bottom-left as iris-radii units) making it scale-invariant — same physical displacement reads identically at any camera distance.',
    tag:         'Module 4',
    tagColour:   'bg-violet-100 text-violet-700',
  },
  {
    key:         'module6_result' as const,
    step:        '06',
    title:       'Hirschberg Angle + Result',
    description: 'Asymmetry score = |L norm − R norm|. Deviation converted via Hirschberg formula: disp_mm × 7°/mm. Banner shows deviation (deg), asymmetry score, condition, ICD-10, and per-eye displacement in iris radii.',
    tag:         'Modules 5 – 7',
    tagColour:   'bg-green-100 text-green-700',
  },
] as const;

export default function ProcessingSteps({ intermediateImages }: ProcessingStepsProps) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 mt-5 print:hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-slate-900 font-semibold text-base">How the AI Calculates This</h2>
          <p className="text-slate-500 text-xs mt-0.5">
            6-step Corneal Light Reflex pipeline — from raw photo to clinical angle
          </p>
        </div>
        <span className="text-xs bg-slate-100 text-slate-500 px-2.5 py-1 rounded-full font-medium">
          Hirschberg Method
        </span>
      </div>

      {/* Steps grid — 2 columns on sm+, 1 on mobile */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {STEPS.map((step) => {
          const imageB64 = intermediateImages[step.key];
          const isWide   = step.key === 'module6_result';   // final image spans full width

          return (
            <div
              key={step.key}
              className={`flex flex-col bg-slate-50 rounded-xl border border-slate-100 overflow-hidden${isWide ? ' sm:col-span-2' : ''}`}
            >
              {/* Image */}
              <div className={`w-full bg-black overflow-hidden flex items-center justify-center${isWide ? ' aspect-[3/1]' : ' aspect-[2/1]'}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/jpeg;base64,${imageB64}`}
                  alt={step.title}
                  className="max-w-full max-h-full object-contain"
                />
              </div>

              {/* Caption */}
              <div className="px-3 py-2.5">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-slate-400 text-xs font-mono font-medium">{step.step}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${step.tagColour}`}>
                    {step.tag}
                  </span>
                </div>
                <h3 className="text-slate-800 text-sm font-semibold mb-0.5">{step.title}</h3>
                <p className="text-slate-500 text-xs leading-relaxed">{step.description}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-slate-100 flex flex-wrap gap-x-5 gap-y-1.5">
        {[
          { colour: 'bg-blue-500',   label: 'Pupil centre' },
          { colour: 'bg-amber-400',  label: 'Corneal light reflex' },
          { colour: 'bg-green-400',  label: 'Iris radius ring' },
          { colour: 'bg-white border border-slate-300', label: 'Displacement vector' },
        ].map(({ colour, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <span className={`w-2.5 h-2.5 rounded-full inline-block flex-shrink-0 ${colour}`} />
            <span className="text-slate-500 text-xs">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
