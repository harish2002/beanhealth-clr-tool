import type { UrgencyTier } from "@/lib/types";
import { URGENCY_CONFIG } from "@/lib/types";

interface UrgencyBadgeProps {
  urgency: UrgencyTier;
  large?:  boolean;
}

export default function UrgencyBadge({ urgency, large = false }: UrgencyBadgeProps) {
  const config = URGENCY_CONFIG[urgency];
  return (
    <span
      className={`inline-flex items-center font-bold tracking-widest uppercase rounded-full
        ${config.badgeColour} text-white
        ${large ? "px-5 py-2 text-sm" : "px-3 py-1 text-xs"}`}
    >
      {config.label}
    </span>
  );
}
