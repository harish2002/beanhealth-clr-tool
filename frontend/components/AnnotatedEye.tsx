"use client";

/**
 * AnnotatedEye
 *
 * Renders the base64-annotated JPEG returned by the API.
 * If no image is available (mock mode or error), shows a placeholder.
 */

interface AnnotatedEyeProps {
  base64Jpeg: string;
  patientName: string;
}

export default function AnnotatedEye({ base64Jpeg, patientName }: AnnotatedEyeProps) {
  if (!base64Jpeg) {
    return (
      <div className="w-full aspect-video bg-slate-800/60 rounded-2xl flex items-center justify-center border border-slate-700/50">
        <div className="text-center text-slate-500">
          <svg className="w-10 h-10 mx-auto mb-2 opacity-40" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
          </svg>
          <p className="text-xs">Annotated image unavailable</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full rounded-2xl overflow-hidden border border-slate-700/50 shadow-lg">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`data:image/jpeg;base64,${base64Jpeg}`}
        alt={`Annotated eye scan for ${patientName}`}
        className="w-full object-contain"
      />
      <div className="px-3 py-2 bg-slate-100 border-t border-slate-200 flex gap-4 text-xs text-slate-500 print:bg-slate-50">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" />
          Pupil centre
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" />
          Light reflex (CLR)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-4 h-0.5 bg-white inline-block" />
          Displacement
        </span>
      </div>
    </div>
  );
}
