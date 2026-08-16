import React from 'react';
import { PatientData } from '../types';
import {
  HeartPulse,
  Activity,
  ArrowUp,
  ArrowDown,
  ArrowRight,
} from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';

interface WardDashboardProps {
  patients: Record<string, PatientData>;
  sparklines: Record<string, number[]>;
  batchedInsights?: Record<string, any>;
  onSelectPatient: (patientId: string) => void;
}

const TrendArrow = ({ trend }: { trend?: string }) => {
  if (!trend) return null;
  if (trend === 'rising') return <ArrowUp className="w-4 h-4 text-[#F46B52]" />;
  if (trend === 'falling') return <ArrowDown className="w-4 h-4 text-[#E9A52F]" />;
  return <ArrowRight className="w-4 h-4 text-slate-300" />;
};

const CESSparkline = ({ data, color }: { data: number[], color: string }) => {
  if (!data || data.length === 0) {
    return <div className="h-8 w-full mt-1 bg-[#F7F7F4] rounded-lg border border-[#DCE4E7] animate-pulse" />;
  }
  const chartData = data.map((val, i) => ({ i, val }));
  const gradId = `grad-ces-${color.replace('#', '')}`;
  return (
    <div className="h-8 w-full mt-1">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.4} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <YAxis domain={['dataMin', 'dataMax']} hide />
          <Area
            type="monotone"
            dataKey="val"
            stroke={color}
            strokeWidth={2}
            fillOpacity={1}
            fill={`url(#${gradId})`}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export const WardDashboard: React.FC<WardDashboardProps> = ({
  patients,
  sparklines,
  batchedInsights,
  onSelectPatient,
}) => {
  const patientList = Object.values(patients);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {patientList.length === 0 ? (
          <div className="col-span-full flex flex-col items-center justify-center p-12 bg-white rounded-2xl border border-slate-200">
            <Activity className="w-12 h-12 text-[#DCE4E7] mb-4" />
            <h3 className="text-lg font-bold text-[#102A43]">No active patients</h3>
            <p className="text-sm text-[#52677D]">Waiting for telemetry data...</p>
          </div>
        ) : (
          patientList.map((patient) => {
            const vitals = patient.vitals || {};
            const hr = vitals.heart_rate;
            const spo2 = vitals.spo2;
            const sys = vitals.bp_systolic;
            const dia = vitals.bp_diastolic;
            const temp = vitals.temperature;
            const ces = patient.composite_entropy || 0;

            const patientInsights = batchedInsights?.[patient.patient_id];
            const narrative = patientInsights?.narrative || patient.narrative_summary;
            const recommendations = patientInsights?.recommendations || patient.recommendations;

            let statusColor = 'bg-[#159A73]';
            let statusText = 'text-[#159A73]';
            let statusBg = 'bg-[#E5F5EF]';
            let riskLabel = 'STABLE';
            if (ces < 0.2) {
              statusColor = 'bg-[#F46B52]';
              statusText = 'text-[#F46B52]';
              statusBg = 'bg-[#FFE0D6]';
              riskLabel = 'CRITICAL';
            } else if (ces < 0.4) {
              statusColor = 'bg-[#E9A52F]';
              statusText = 'text-[#E9A52F]';
              statusBg = 'bg-[#F8E6A7]';
              riskLabel = 'WARNING';
            } else if (ces < 0.6) {
              statusColor = 'bg-[#F4D03F]';
              statusText = 'text-[#F4D03F]';
              statusBg = 'bg-[#FEF9E7]';
              riskLabel = 'WATCH';
            }

            return (
              <div
                key={patient.patient_id}
                onClick={() => onSelectPatient(patient.patient_id)}
                className="p-5 rounded-2xl bg-white border border-[#DCE4E7] flex flex-col justify-between hover:shadow-md transition-all group relative overflow-hidden shadow-2xs cursor-pointer"
              >
                <div className="flex items-center justify-between mb-4 gap-2">
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    <div className={`w-10 h-10 shrink-0 rounded-xl ${statusBg} flex items-center justify-center ${statusText} shadow-2xs`}>
                      <HeartPulse className="w-5 h-5" />
                    </div>
                    <div className="min-w-0">
                      <h3 className="font-bold text-[#102A43] truncate" title={patient.name || patient.patient_id}>{patient.name || patient.patient_id}</h3>
                      <p className="text-xs text-[#52677D] font-mono truncate" title={patient.patient_id}>ID: {patient.patient_id}</p>
                    </div>
                  </div>
                  <div className={`shrink-0 px-2.5 py-1 rounded border ${statusText} ${statusBg} border-current border-opacity-20 flex items-center gap-1.5`}>
                    <span className={`w-2 h-2 rounded-full ${statusColor} animate-pulse shrink-0`} />
                    <span className="text-xs font-bold whitespace-nowrap">CES: {ces.toFixed(2)}</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="p-3 bg-[#F7F7F4] rounded-xl border border-[#DCE4E7] flex flex-col justify-between">
                    <div className="text-[10px] font-bold text-[#52677D] uppercase">Heart Rate</div>
                    <div className="flex items-end justify-between mt-1">
                      <div className="text-xl font-bold text-[#102A43]">{hr?.value ?? '--'} <span className="text-xs font-normal text-[#52677D]">bpm</span></div>
                      <TrendArrow trend={hr?.trend} />
                    </div>
                  </div>
                  <div className="p-3 bg-[#F7F7F4] rounded-xl border border-[#DCE4E7] flex flex-col justify-between">
                    <div className="text-[10px] font-bold text-[#52677D] uppercase">SpO2</div>
                    <div className="flex items-end justify-between mt-1">
                      <div className="text-xl font-bold text-[#102A43]">{spo2?.value ?? '--'} <span className="text-xs font-normal text-[#52677D]">%</span></div>
                      <TrendArrow trend={spo2?.trend} />
                    </div>
                  </div>
                  <div className="p-3 bg-[#F7F7F4] rounded-xl border border-[#DCE4E7] flex flex-col justify-between">
                    <div className="text-[10px] font-bold text-[#52677D] uppercase">Blood Pressure</div>
                    <div className="flex items-end justify-between mt-1">
                      <div className="text-xl font-bold text-[#102A43]">{sys?.value ?? '--'}/{dia?.value ?? '--'}</div>
                      <TrendArrow trend={sys?.trend} />
                    </div>
                  </div>
                  <div className="p-3 bg-[#F7F7F4] rounded-xl border border-[#DCE4E7] flex flex-col justify-between">
                    <div className="text-[10px] font-bold text-[#52677D] uppercase">Temperature</div>
                    <div className="flex items-end justify-between mt-1">
                      <div className="text-xl font-bold text-[#102A43]">{temp?.value ? temp.value.toFixed(1) : '--'} <span className="text-xs font-normal text-[#52677D]">°C</span></div>
                      <TrendArrow trend={temp?.trend} />
                    </div>
                  </div>
                </div>

                {/* Micro-visualization: CES Sparkline */}
                <div className="pt-3 border-t border-[#DCE4E7]">
                  <div className="flex items-center justify-between text-[10px] font-mono text-[#52677D]">
                    <span>Risk Trend (CES):</span>
                    <strong className="text-[#102A43]">{riskLabel}</strong>
                  </div>
                  <CESSparkline 
                    data={sparklines[patient.patient_id]} 
                    color={statusColor.replace('bg-', '').replace('[', '').replace(']', '')} 
                  />
                </div>

                {/* CHRONOS Agent Sub-Panels */}
                <div className="mt-4 flex flex-col gap-3">
                  
                  {/* Pharmacology Context Agent */}
                  {patient.active_drugs && patient.active_drugs.length > 0 && (
                    <div className="bg-[#F7F7F4] p-3 rounded-xl border border-[#DCE4E7]">
                      <div className="text-[10px] font-bold text-[#52677D] uppercase mb-1">Pharmacology Context Agent</div>
                      <div className="flex items-center gap-2 flex-wrap">
                        {patient.active_drugs.map((drug, idx) => (
                          <div key={idx} className="text-xs bg-white border border-[#DCE4E7] px-2 py-1 rounded shadow-2xs text-[#102A43] font-medium">
                            {drug.drug_name} {drug.dose && <span className="text-[#52677D]">{parseFloat(Number(drug.dose).toFixed(3))} {drug.unit}</span>}
                          </div>
                        ))}
                        {patient.drug_masked && (
                          <div className="text-xs bg-[#FFE0D6] text-[#F46B52] border border-[#F46B52]/20 px-2 py-1 rounded shadow-2xs font-bold animate-pulse">
                            Drug Masking Detected
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Narrative Agent */}
                  {narrative && (
                    <div className="bg-white p-3 rounded-xl border border-[#DCE4E7] shadow-2xs">
                      <div className="text-[10px] font-bold text-[#52677D] uppercase mb-1">Narrative Agent</div>
                      <p className="text-xs text-[#102A43] leading-relaxed">
                        {narrative}
                      </p>
                    </div>
                  )}

                  {/* Clinical Reasoning Agent */}
                  {recommendations && recommendations.length > 0 && (
                    <div className="bg-[#E5F5EF] p-3 rounded-xl border border-[#159A73]/20">
                      <div className="text-[10px] font-bold text-[#159A73] uppercase mb-1">Clinical Reasoning Agent</div>
                      {recommendations.map((rec: any, idx: number) => (
                        <div key={idx} className="text-xs text-[#102A43]">
                          <div className="font-bold mb-0.5">{rec.intervention} <span className="text-[#159A73]">({rec.success_rate}% success)</span></div>
                          <div className="text-[#52677D] italic">{rec.evidence_summary}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
