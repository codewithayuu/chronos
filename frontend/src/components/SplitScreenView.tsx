import React, { useState, useEffect } from 'react';
import { PatientData } from '../types';
import { ArrowLeft, HeartPulse, AlertTriangle, CheckCircle, Activity, Heart, Wind, Thermometer, BrainCircuit, Loader2, Sparkles, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { VitalChart } from './VitalChart';
import { CESGauge } from './CESGauge';
import { EntropyBars } from './EntropyBars';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';

interface SplitScreenViewProps {
  patient: PatientData;
  onBack: () => void;
}

const TrendArrow = ({ trend }: { trend?: string }) => {
  if (!trend) return null;
  if (trend === 'rising') return <span className="text-[#F46B52] ml-1">↑</span>;
  if (trend === 'falling') return <span className="text-[#E9A52F] ml-1">↓</span>;
  return <span className="text-slate-400 ml-1">→</span>;
};

const MiniVitalChart = ({ data, dataKey, color }: { data: any[], dataKey: string, color: string }) => (
  <div className="h-14 w-full mt-2">
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <defs>
          <linearGradient id={`grad-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <YAxis domain={['auto', 'auto']} hide />
        <Area
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={2}
          fillOpacity={1}
          fill={`url(#grad-${dataKey})`}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  </div>
);

export const SplitScreenView: React.FC<SplitScreenViewProps> = ({ patient, onBack }) => {
  const [isConsulting, setIsConsulting] = useState(false);
  const [liveAgentData, setLiveAgentData] = useState<{ recommendations?: any[], narrative?: string } | null>(null);
  const [consultError, setConsultError] = useState<string | null>(null);
  const [autoFetchedFor, setAutoFetchedFor] = useState<string | null>(null);

  const vitals = patient.vitals || {};
  const ces = patient.composite_entropy || 0;

  const handleConsultGemini = async () => {
    setIsConsulting(true);
    setConsultError(null);
    try {
      const res = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to fetch CHRONOS AI recommendation');
      
      let jsonText = data.result;
      jsonText = jsonText.replace(/```json/gi, '').replace(/```/g, '').trim();
      const parsed = JSON.parse(jsonText);
      setLiveAgentData(parsed);
    } catch (err: any) {
      setConsultError(err.message);
    } finally {
      setIsConsulting(false);
    }
  };

  useEffect(() => {
    if (patient.patient_id !== autoFetchedFor) {
      setAutoFetchedFor(patient.patient_id);
      handleConsultGemini();
    }
  }, [patient.patient_id, autoFetchedFor]);

  // Derive status
  let statusColor = 'bg-[#159A73]';
  let statusText = 'text-[#159A73]';
  let statusBorder = 'border-[#159A73]';
  let riskLabel = 'STABLE';
  if (ces < 0.2) {
    statusColor = 'bg-[#F46B52]';
    statusText = 'text-[#F46B52]';
    statusBorder = 'border-[#F46B52]';
    riskLabel = 'CRITICAL';
  } else if (ces < 0.4) {
    statusColor = 'bg-[#E9A52F]';
    statusText = 'text-[#E9A52F]';
    statusBorder = 'border-[#E9A52F]';
    riskLabel = 'WARNING';
  } else if (ces < 0.6) {
    statusColor = 'bg-[#F4D03F]';
    statusText = 'text-[#F4D03F]';
    statusBorder = 'border-[#F4D03F]';
    riskLabel = 'WATCH';
  }
  const colorHex = statusColor.replace('bg-', '').replace('[', '').replace(']', '');

  // Count traditional alarms (mock logic)
  let traditionalAlarmCount = 0;
  const hr = vitals.heart_rate?.value;
  if (hr && (hr < 50 || hr > 120)) traditionalAlarmCount++;
  const spo2 = vitals.spo2?.value;
  if (spo2 && spo2 < 90) traditionalAlarmCount++;
  const sys = vitals.bp_systolic?.value;
  if (sys && (sys < 90 || sys > 180)) traditionalAlarmCount++;

  // Mock time-series data for VitalChart
  const mockTimeSeries = React.useMemo(() => {
    const data = [];
    const now = Date.now();
    let currentHR = vitals.heart_rate?.value || 80;
    let currentSpO2 = vitals.spo2?.value || 98;
    let currentSys = vitals.bp_systolic?.value || 120;
    let currentTemp = vitals.temperature?.value || 37.0;
    let currentEnt = ces || 0.5;
    for (let i = 0; i < 60; i++) {
      data.push({
        time: new Date(now - (60 - i) * 60000).toISOString(),
        hr: currentHR + (Math.random() * 4 - 2),
        spo2: currentSpO2 + (Math.random() * 2 - 1),
        sys: currentSys + (Math.random() * 6 - 3),
        temp: currentTemp + (Math.random() * 0.4 - 0.2),
        entropy: currentEnt + (Math.random() * 0.05 - 0.025)
      });
      currentHR += (Math.random() * 2 - 1);
      currentSpO2 = Math.min(100, currentSpO2 + (Math.random() * 0.5 - 0.25));
      currentSys += (Math.random() * 2 - 1);
      currentTemp += (Math.random() * 0.1 - 0.05);
      currentEnt += (Math.random() * 0.02 - 0.01);
    }
    return data;
  }, [vitals, ces]);

  return (
    <div className="flex flex-col gap-6 animate-in fade-in zoom-in-95 duration-300 relative">
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-[#52677D] hover:text-[#102A43] transition-colors bg-white border border-[#DCE4E7] px-4 py-2 rounded-xl shadow-2xs font-bold text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Ward
        </button>
        <div className="flex items-center gap-3 min-w-0 flex-1 justify-end ml-4">
          <h2 className="text-xl font-bold text-[#102A43] truncate" title={patient.name || patient.patient_id}>{patient.name || patient.patient_id}</h2>
          <span className={`shrink-0 px-3 py-1 rounded border ${statusBorder}/20 ${statusText} font-bold text-xs shadow-2xs bg-white whitespace-nowrap`}>
            {riskLabel} RISK
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        {/* LEFT SIDE: TRADITIONAL MONITOR */}
        <div className="bg-white rounded-2xl border border-[#DCE4E7] shadow-2xs flex flex-col overflow-hidden">
          <div className="bg-[#F7F7F4] border-b border-[#DCE4E7] px-6 py-4 flex justify-between items-center">
            <div>
              <h3 className="font-bold text-[#102A43] uppercase text-xs tracking-wider">Advanced Telemetry</h3>
              <p className="text-[11px] text-[#52677D]">High-resolution vital sign traces</p>
            </div>
            <div className={`px-2.5 py-1 rounded border flex items-center gap-1.5 font-bold text-xs ${traditionalAlarmCount > 0 ? 'bg-[#FFE0D6] border-[#F46B52]/20 text-[#F46B52]' : 'bg-[#E5F5EF] border-[#159A73]/20 text-[#159A73]'}`}>
              {traditionalAlarmCount > 0 ? <AlertTriangle className="w-3 h-3 animate-pulse" /> : <CheckCircle className="w-3 h-3" />}
              {traditionalAlarmCount > 0 ? `${traditionalAlarmCount} ALARMS` : 'NORMAL'}
            </div>
          </div>

          <div className="p-6 flex-1 flex flex-col">
            <div className="space-y-6 flex-1">
              {/* HR */}
              <div className="border-b border-[#DCE4E7] pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[#F46B52]">
                    <Heart className="w-5 h-5 opacity-80" />
                    <span className="font-bold text-[#102A43]">Heart Rate</span>
                  </div>
                  <div className="text-2xl font-black text-[#102A43] tabular-nums tracking-tight">
                    {vitals.heart_rate?.value ?? '--'} <span className="text-sm font-semibold text-[#52677D]">bpm</span>
                  </div>
                </div>
                <MiniVitalChart data={mockTimeSeries} dataKey="hr" color="#F46B52" />
              </div>
              
              {/* SpO2 */}
              <div className="border-b border-[#DCE4E7] pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[#4FA8B8]">
                    <Wind className="w-5 h-5 opacity-80" />
                    <span className="font-bold text-[#102A43]">SpO2</span>
                  </div>
                  <div className="text-2xl font-black text-[#102A43] tabular-nums tracking-tight">
                    {vitals.spo2?.value ?? '--'} <span className="text-sm font-semibold text-[#52677D]">%</span>
                  </div>
                </div>
                <MiniVitalChart data={mockTimeSeries} dataKey="spo2" color="#4FA8B8" />
              </div>

              {/* BP */}
              <div className="border-b border-[#DCE4E7] pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[#8B5CF6]">
                    <Activity className="w-5 h-5 opacity-80" />
                    <span className="font-bold text-[#102A43]">Blood Pressure</span>
                  </div>
                  <div className="text-2xl font-black text-[#102A43] tabular-nums tracking-tight">
                    {vitals.bp_systolic?.value ?? '--'}/{vitals.bp_diastolic?.value ?? '--'} <span className="text-sm font-semibold text-[#52677D]">mmHg</span>
                  </div>
                </div>
                <MiniVitalChart data={mockTimeSeries} dataKey="sys" color="#8B5CF6" />
              </div>

              {/* Temp */}
              <div className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 text-[#F59E0B]">
                    <Thermometer className="w-5 h-5 opacity-80" />
                    <span className="font-bold text-[#102A43]">Temperature</span>
                  </div>
                  <div className="text-2xl font-black text-[#102A43] tabular-nums tracking-tight">
                    {vitals.temperature?.value ?? '--'} <span className="text-sm font-semibold text-[#52677D]">°C</span>
                  </div>
                </div>
                <MiniVitalChart data={mockTimeSeries} dataKey="temp" color="#F59E0B" />
              </div>
            </div>

            {/* EMBEDDED AI AGENT (Moved from floating overlay) */}
            <div className="mt-6 border border-[#4FA8B8] rounded-2xl overflow-hidden shadow-2xs bg-white flex flex-col relative">
              <div className="bg-gradient-to-r from-[#4FA8B8] to-[#2c7785] p-3 flex items-center justify-between shrink-0 shadow-sm">
                <div className="flex items-center gap-2 text-white">
                  <Sparkles className="w-4 h-4 animate-pulse" />
                  <span className="font-bold text-xs tracking-wider">CHRONOS AI AGENT</span>
                </div>
              </div>
              
              <div className="p-5 bg-[#F7F7F4]/50">
                {isConsulting ? (
                  <div className="flex flex-col items-center justify-center py-8 space-y-4">
                    <Loader2 className="w-6 h-6 text-[#4FA8B8] animate-spin" />
                    <p className="text-xs font-semibold text-[#52677D] animate-pulse">Analyzing telemetry & establishing baseline...</p>
                  </div>
                ) : consultError ? (
                  <div className="text-[#F46B52] bg-[#FFE0D6] p-3 rounded-xl text-xs font-bold border border-[#F46B52]/20">
                    {consultError}
                  </div>
                ) : liveAgentData ? (
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-[9px] uppercase tracking-wider font-bold text-[#4FA8B8] mb-1">Live Assessment</h4>
                      <p className="text-xs text-[#102A43] leading-relaxed font-medium">
                        {liveAgentData.narrative}
                      </p>
                    </div>
                    
                    {liveAgentData.recommendations && liveAgentData.recommendations.length > 0 && (
                      <div>
                        <h4 className="text-[9px] uppercase tracking-wider font-bold text-[#4FA8B8] mb-2 mt-4">Top Recommendations</h4>
                        <div className="space-y-2">
                          {liveAgentData.recommendations.map((rec: any, i: number) => (
                            <div key={i} className="bg-white p-3 rounded-xl border border-[#DCE4E7]">
                              <div className="flex justify-between items-center mb-1">
                                <span className="font-bold text-[#102A43] text-xs">{rec.intervention}</span>
                                <span className="text-[9px] font-bold text-[#159A73] bg-[#E5F5EF] px-1.5 py-0.5 rounded border border-[#159A73]/20">
                                  {rec.success_rate}% success
                                </span>
                              </div>
                              <p className="text-[10px] text-[#52677D]">{rec.evidence_summary}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8 text-xs text-[#52677D]">
                    Waiting for telemetry data...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT SIDE: CHRONOS INTELLIGENCE */}
        <div className="bg-white rounded-2xl border border-[#DCE4E7] shadow-2xs flex flex-col overflow-hidden relative">
          {/* Subtle colored glow at top based on risk */}
          <div className={`absolute top-0 left-0 right-0 h-1 ${statusColor}`} />
          
          <div className="bg-white border-b border-[#DCE4E7] px-6 py-4 flex items-center justify-between mt-1">
            <div>
              <h3 className="font-bold text-[#102A43] uppercase text-xs tracking-wider flex items-center gap-2">
                <HeartPulse className="w-4 h-4 text-[#4FA8B8]" />
                CHRONOS Intelligence
              </h3>
              <p className="text-[11px] text-[#52677D]">Multi-Agent Entropy Analysis</p>
            </div>
            <button 
              onClick={() => {
                handleConsultGemini();
              }}
              disabled={isConsulting}
              className="flex items-center gap-2 px-3 py-1.5 bg-[#E5F5EF] text-[#159A73] border border-[#159A73]/20 hover:bg-[#159A73] hover:text-white rounded-xl font-bold text-xs transition-all shadow-sm"
            >
              {isConsulting ? <Loader2 className="w-3 h-3 animate-spin" /> : <BrainCircuit className="w-3 h-3" />}
              {isConsulting ? 'Consulting...' : 'Force AI Consult'}
            </button>
          </div>

          <div className="p-6 flex-1 flex flex-col gap-6 bg-[#F7F7F4]/30 overflow-y-auto">
            
            {/* CES Visualizers */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <CESGauge 
                value={ces} 
                rawValue={patient.drug_masked ? ces + 0.15 : null} 
                severityLabel={riskLabel} 
                color={colorHex} 
              />
              <EntropyBars 
                vitals={vitals} 
                severityColor={colorHex} 
                contributingVitals={ces < 0.4 ? ['heart_rate', 'bp_systolic'] : []}
              />
            </div>

            {/* Vital Chart with Entropy Overlay */}
            <div className="w-full">
              <VitalChart
                data={mockTimeSeries}
                valueKey="hr"
                entropyKey="entropy"
                label="Heart Rate vs Entropy"
                unit="bpm"
                icon={<Heart className="w-4 h-4" />}
                thresholdLow={50}
                thresholdHigh={120}
                currentValue={vitals.heart_rate?.value}
                currentTrend={vitals.heart_rate?.trend}
                severityColor={colorHex}
                isContributing={ces < 0.4}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="flex flex-col gap-6">
                {/* Narrative Agent */}
                <div className={`bg-white border ${liveAgentData ? 'border-[#4FA8B8] ring-1 ring-[#4FA8B8]' : 'border-[#DCE4E7]'} p-4 rounded-xl shadow-2xs h-full relative`}>
                  {liveAgentData && (
                    <div className="absolute top-0 right-0 bg-[#4FA8B8] text-white text-[9px] uppercase font-bold px-2 py-0.5 rounded-bl-lg rounded-tr-xl">
                      Live CHRONOS AI
                    </div>
                  )}
                  <h4 className="text-[10px] uppercase tracking-wider font-bold text-[#52677D] mb-2">Narrative Agent</h4>
                  <p className="text-sm text-[#102A43] leading-relaxed">
                    {liveAgentData?.narrative || patient.narrative_summary || "No narrative generated yet. Establishing baseline complexity..."}
                  </p>
                </div>

                {/* Pharmacology Agent */}
                <div className="bg-white border border-[#DCE4E7] p-4 rounded-xl shadow-2xs">
                  <h4 className="text-[10px] uppercase tracking-wider font-bold text-[#52677D] mb-2">Pharmacology Context Agent</h4>
                  {patient.active_drugs && patient.active_drugs.length > 0 ? (
                    <div className="flex flex-col gap-2">
                      <div className="flex flex-wrap gap-2">
                        {patient.active_drugs.map((drug, i) => (
                          <span key={i} className="px-3 py-1.5 bg-[#F7F7F4] border border-[#DCE4E7] rounded-lg text-xs font-bold text-[#102A43]">
                            {drug.drug_name} <span className="text-[#52677D] font-normal">{drug.dose} {drug.unit}</span>
                          </span>
                        ))}
                      </div>
                      {patient.drug_masked && (
                        <div className="mt-2 bg-[#FFE0D6] border border-[#F46B52]/20 text-[#F46B52] px-3 py-2 rounded-lg text-xs font-bold flex items-center gap-2 animate-pulse">
                          <AlertTriangle className="w-4 h-4" />
                          Warning: Vitals artificially supported by active drugs. Underlying complexity is critical.
                        </div>
                      )}
                    </div>
                  ) : (
                    <span className="text-xs text-[#52677D]">No active infusions impacting physiology.</span>
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-6">
                {/* Clinical Reasoning Agent */}
                <div className={`bg-white border ${liveAgentData ? 'border-[#4FA8B8] ring-1 ring-[#4FA8B8]' : 'border-[#DCE4E7]'} p-4 rounded-xl shadow-2xs h-full relative`}>
                  {liveAgentData && (
                    <div className="absolute top-0 right-0 bg-[#4FA8B8] text-white text-[9px] uppercase font-bold px-2 py-0.5 rounded-bl-lg rounded-tr-xl">
                      Live CHRONOS AI
                    </div>
                  )}
                  <h4 className="text-[10px] uppercase tracking-wider font-bold text-[#52677D] mb-2">Clinical Reasoning Agent</h4>
                  {(liveAgentData?.recommendations || patient.recommendations)?.length ? (
                    <div className="space-y-3">
                      {(liveAgentData?.recommendations || patient.recommendations || []).map((rec: any, i: number) => (
                        <div key={i} className="flex gap-3 items-start border-l-2 border-[#4FA8B8] pl-3">
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <span className="font-bold text-[#102A43] text-sm">{rec.intervention}</span>
                              <span className="text-xs font-bold bg-[#E5F5EF] text-[#159A73] px-2 py-0.5 rounded border border-[#159A73]/20">
                                {rec.success_rate}% historical success
                              </span>
                            </div>
                            <p className="text-xs text-[#52677D] italic">{rec.evidence_summary}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <span className="text-xs text-[#52677D]">No acute interventions recommended at this time.</span>
                  )}
                </div>
              </div>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
};
