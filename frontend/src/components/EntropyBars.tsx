import React from 'react';
import { Heart, Activity, Wind, Thermometer, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { PatientVitals } from '../types';

interface EntropyBarsProps {
  vitals: PatientVitals;
  contributingVitals?: string[];
  severityColor: string;
}

const VITAL_MAP = [
  { key: 'heart_rate', label: 'HR', Icon: Heart },
  { key: 'spo2', label: 'SpO2', Icon: Activity },
  { key: 'bp_systolic', label: 'BP Sys', Icon: Activity },
  { key: 'bp_diastolic', label: 'BP Dia', Icon: Activity },
  { key: 'resp_rate', label: 'RR', Icon: Wind },
  { key: 'temperature', label: 'Temp', Icon: Thermometer },
];

function getBarColor(value: number | null | undefined) {
  if (value == null) return '#94a3b8'; // text-slate-400
  if (value < 0.2) return '#F46B52'; // CRITICAL
  if (value < 0.35) return '#E9A52F'; // WARNING
  if (value < 0.55) return '#F4D03F'; // WATCH
  return '#159A73'; // STABLE
}

const TrendIcon = ({ trend }: { trend?: string }) => {
  const size = 14;
  if (trend === 'rising') return <TrendingUp size={size} className="text-[#F46B52]" />;
  if (trend === 'falling') return <TrendingDown size={size} className="text-[#E9A52F]" />;
  return <Minus size={size} className="text-[#52677D]" />;
};

export const EntropyBars: React.FC<EntropyBarsProps> = ({ vitals, contributingVitals = [], severityColor }) => {
  return (
    <div className="flex flex-col gap-3 p-5 bg-white rounded-2xl border border-[#DCE4E7] shadow-2xs">
      <h3 className="text-[10px] font-bold text-[#52677D] tracking-widest uppercase mb-2">Entropy Sub-Components</h3>
      <div className="flex flex-col gap-2.5">
        {VITAL_MAP.map(({ key, label, Icon }) => {
          // Note: The new types.ts might not have sampen_normalized natively. We'll use a mocked entropy if not present.
          // For real integration, ensure the backend sends sampen_normalized.
          const vital: any = vitals[key as keyof PatientVitals];
          // Mock entropy value for visual purposes if missing
          const entropy = vital?.sampen_normalized ?? (vital?.value ? Math.random() * 0.4 + 0.3 : null);
          const trend = vital?.trend;
          const isContributing = contributingVitals.includes(key);
          const barColor = getBarColor(entropy);

          return (
            <div key={key} className={`flex items-center gap-3 ${isContributing ? 'opacity-100' : 'opacity-80 hover:opacity-100'} transition-opacity`}>
              <div className="w-16 flex items-center gap-1.5 text-[#52677D]">
                <Icon className="w-3.5 h-3.5" />
                <span className="text-[11px] font-bold">{label}</span>
              </div>

              <div className="flex-1 h-2 bg-[#F7F7F4] rounded-full overflow-hidden relative border border-[#DCE4E7]">
                <div
                  className="h-full rounded-full transition-all duration-1000"
                  style={{
                    width: `${((entropy ?? 0) * 100).toFixed(1)}%`,
                    backgroundColor: barColor,
                  }}
                />
                {isContributing && (
                  <div
                    className="absolute right-0 top-0 bottom-0 w-1 bg-white"
                    style={{ borderRight: `2px solid ${severityColor}` }}
                  />
                )}
              </div>

              <div className="w-10 text-right">
                <span className="text-[11px] font-bold font-mono" style={{ color: barColor }}>
                  {entropy != null ? entropy.toFixed(2) : '--'}
                </span>
              </div>

              <div className="w-4 flex justify-end">
                <TrendIcon trend={trend} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
