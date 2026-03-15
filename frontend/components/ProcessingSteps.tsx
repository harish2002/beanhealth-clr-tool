import Image from 'next/image';
import type { SuccessResponse } from '@/lib/types';

interface ProcessingStepsProps {
  intermediateImages: NonNullable<SuccessResponse['intermediate_images']>;
}

export default function ProcessingSteps({ intermediateImages }: ProcessingStepsProps) {
  const steps = [
    {
      title: '1. Eye Detection',
      description: 'Face located and eye regions extracted.',
      imageB64: intermediateImages.module1_crops,
    },
    {
      title: '2. Pupil Center',
      description: 'Center of both pupils localized.',
      imageB64: intermediateImages.module2_pupil,
    },
    {
      title: '3. Light Reflex',
      description: 'Corneal light reflex (flash) identified.',
      imageB64: intermediateImages.module3_clr,
    },
    {
      title: '4. Displacement Vector',
      description: 'Distance between pupil and reflex measured.',
      imageB64: intermediateImages.module4_vector,
    },
  ];

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 mt-5">
      <h2 className="text-slate-900 font-semibold mb-4">How AI Calculates This</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {steps.map((step, index) => (
          <div key={index} className="bg-slate-50 rounded-xl p-3 border border-slate-100 flex flex-col">
            <h3 className="text-slate-800 text-sm font-medium mb-1">{step.title}</h3>
            <p className="text-slate-500 text-xs mb-3 flex-grow">{step.description}</p>
            <div className="relative w-full aspect-[2/1] rounded bg-black overflow-hidden flex items-center justify-center">
              {/* Using a standard img tag with pure base64 jpeg data */}
              <img
                src={`data:image/jpeg;base64,${step.imageB64}`}
                alt={step.title}
                className="max-w-full max-h-full object-contain"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
