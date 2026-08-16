import React, { useState } from 'react';
import { PatientData, AlertData } from '../types';
import { Sparkles, Loader2, BrainCircuit, Users, HeartPulse, Activity } from 'lucide-react';

interface AnalyticsDashboardProps {
  patients: Record<string, PatientData>;
  alerts: AlertData[];
}

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ patients, alerts }) => {
  const [isConsulting, setIsConsulting] = useState(false);
  const [wardAgentData, setWardAgentData] = useState<any | null>(null);
  const [consultError, setConsultError] = useState<string | null>(null);

  const patientList = Object.values(patients);
  const totalPatients = patientList.length;
  
  let criticalCount = 0;
  let warningCount = 0;
  let stableCount = 0;

  patientList.forEach(p => {
    const ces = p.composite_entropy || 0;
    if (ces < 0.2) criticalCount++;
    else if (ces < 0.4) warningCount++;
    else stableCount++;
  });

  const activeAlertsCount = alerts.filter(a => !a.acknowledged).length;

  const handleConsultSupervisor = async () => {
    setIsConsulting(true);
    setConsultError(null);
    try {
      const res = await fetch('/api/gemini/ward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patients, alerts })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to fetch CHRONOS Ward Supervisor analysis');
      
      let jsonText = data.result;
      jsonText = jsonText.replace(/```json/gi, '').replace(/```/g, '').trim();
      const parsed = JSON.parse(jsonText);
      setWardAgentData(parsed);
    } catch (err: any) {
      setConsultError(err.message);
    } finally {
      setIsConsulting(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 animate-in fade-in zoom-in-95 duration-300">
      
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-2xl border border-[#DCE4E7] shadow-2xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#E5F5EF] text-[#159A73] flex items-center justify-center">
            <Users className="w-6 h-6" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#52677D] uppercase tracking-wider">Total Patients</p>
            <p className="text-2xl font-black text-[#102A43]">{totalPatients}</p>
          </div>
        </div>
        
        <div className="bg-white p-5 rounded-2xl border border-[#DCE4E7] shadow-2xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#FFE0D6] text-[#F46B52] flex items-center justify-center">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#52677D] uppercase tracking-wider">Critical (CES &lt; 0.2)</p>
            <p className="text-2xl font-black text-[#102A43]">{criticalCount}</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-[#DCE4E7] shadow-2xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#F8E6A7] text-[#E9A52F] flex items-center justify-center">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#52677D] uppercase tracking-wider">Warning (CES &lt; 0.4)</p>
            <p className="text-2xl font-black text-[#102A43]">{warningCount}</p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-[#DCE4E7] shadow-2xs flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-[#FFE0D6] text-[#F46B52] flex items-center justify-center">
            <HeartPulse className="w-6 h-6" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#52677D] uppercase tracking-wider">Active Alerts</p>
            <p className="text-2xl font-black text-[#102A43]">{activeAlertsCount}</p>
          </div>
        </div>
      </div>

      {/* CHRONOS WARD SUPERVISOR AI */}
      <div className="bg-white rounded-2xl border border-[#4FA8B8] shadow-2xs flex flex-col overflow-hidden relative">
        <div className="bg-gradient-to-r from-[#4FA8B8] to-[#2c7785] px-6 py-4 flex items-center justify-between mt-1 shrink-0">
          <div>
            <h3 className="font-bold text-white uppercase text-sm tracking-wider flex items-center gap-2">
              <Sparkles className="w-5 h-5 animate-pulse" />
              CHRONOS Ward Supervisor AI
            </h3>
            <p className="text-[11px] text-white/80 mt-0.5">Macro-level physiological complexity analysis</p>
          </div>
          <button 
            onClick={handleConsultSupervisor}
            disabled={isConsulting}
            className="flex items-center gap-2 px-4 py-2 bg-white text-[#4FA8B8] hover:bg-[#F7F7F4] rounded-xl font-bold text-xs transition-all shadow-sm cursor-pointer"
          >
            {isConsulting ? <Loader2 className="w-4 h-4 animate-spin" /> : <BrainCircuit className="w-4 h-4" />}
            {isConsulting ? 'Running Unit Analysis...' : 'Run Unit Analysis'}
          </button>
        </div>

        <div className="p-6 md:p-8 flex-1 bg-[#F7F7F4]/30 min-h-[400px]">
          {isConsulting ? (
            <div className="flex flex-col items-center justify-center py-20 space-y-6">
              <div className="relative">
                <Loader2 className="w-12 h-12 text-[#4FA8B8] animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-[#2c7785] animate-pulse" />
                </div>
              </div>
              <p className="text-sm font-bold text-[#52677D] animate-pulse tracking-wide">Aggregating telemetry streams and generating macro insights...</p>
            </div>
          ) : consultError ? (
            <div className="text-[#F46B52] bg-[#FFE0D6] p-4 rounded-xl text-sm font-bold border border-[#F46B52]/20 flex items-center justify-center h-full">
              {consultError}
            </div>
          ) : wardAgentData ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-8">
                <div>
                  <h4 className="text-xs uppercase tracking-widest font-bold text-[#4FA8B8] mb-3 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4FA8B8]"></span>
                    Ward Status Summary
                  </h4>
                  <p className="text-base text-[#102A43] leading-relaxed bg-white p-4 rounded-xl border border-[#DCE4E7] shadow-sm">
                    {wardAgentData.ward_status_summary}
                  </p>
                </div>

                <div>
                  <h4 className="text-xs uppercase tracking-widest font-bold text-[#4FA8B8] mb-3 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4FA8B8]"></span>
                    Systemic Patterns
                  </h4>
                  <p className="text-sm text-[#102A43] leading-relaxed bg-white p-4 rounded-xl border border-[#DCE4E7] shadow-sm italic">
                    {wardAgentData.systemic_patterns}
                  </p>
                </div>
                
                <div>
                  <h4 className="text-xs uppercase tracking-widest font-bold text-[#4FA8B8] mb-3 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4FA8B8]"></span>
                    Resource Recommendations
                  </h4>
                  <p className="text-sm text-[#102A43] leading-relaxed bg-white p-4 rounded-xl border border-[#DCE4E7] shadow-sm font-medium">
                    {wardAgentData.resource_recommendations}
                  </p>
                </div>
              </div>

              <div>
                <h4 className="text-xs uppercase tracking-widest font-bold text-[#4FA8B8] mb-3 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#F46B52] animate-pulse"></span>
                  Critical Patient Focus
                </h4>
                {wardAgentData.critical_focus && wardAgentData.critical_focus.length > 0 ? (
                  <div className="space-y-3">
                    {wardAgentData.critical_focus.map((focus: any, i: number) => (
                      <div key={i} className="bg-white p-4 rounded-xl border-l-4 border-l-[#F46B52] border-y border-r border-[#DCE4E7] shadow-sm">
                        <span className="font-bold text-[#102A43] text-sm block mb-1">{focus.patient_id}</span>
                        <p className="text-xs text-[#52677D]">{focus.reason}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-[#E5F5EF] border border-[#159A73]/20 text-[#159A73] p-4 rounded-xl text-sm font-bold shadow-sm">
                    No critical patients requiring immediate supervisor override.
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20 opacity-50 h-full">
              <BrainCircuit className="w-16 h-16 text-[#DCE4E7] mb-4" />
              <p className="text-sm text-[#52677D] font-medium">Supervisor AI is idle. Click 'Run Unit Analysis' to begin.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
