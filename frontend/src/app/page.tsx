"use client";
import React, { useState } from 'react';
import { Sidebar, NavTabId } from '../components/Sidebar';
import { Header } from '../components/Header';
import { WardDashboard } from '../components/WardDashboard';
import { SplitScreenView } from '../components/SplitScreenView';
import { AlertFeed } from '../components/AlertFeed';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { useChronosWebSocket } from '../hooks/useChronosWebSocket';

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTabId>('ward');
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [mounted, setMounted] = useState(false);
  const [batchedInsights, setBatchedInsights] = useState<Record<string, any>>({});
  const [isBatchFetching, setIsBatchFetching] = useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  const {
    patients,
    alerts,
    connected,
    systemStatus,
    sparklines,
    acknowledgeAlert,
  } = useChronosWebSocket();

  const activeAlertCount = alerts.filter(a => !a.acknowledged).length;
  const patientCount = Object.keys(patients).length;

  const handleRefreshAllInsights = async () => {
    setIsBatchFetching(true);
    try {
      const patientList = Object.values(patients);
      const chunkSize = 10;
      const updatedInsights = { ...batchedInsights };
      
      for (let i = 0; i < patientList.length; i += chunkSize) {
        const chunk = patientList.slice(i, i + chunkSize);
        const res = await fetch('/api/gemini/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patients: chunk })
        });
        
        if (res.ok) {
          const data = await res.json();
          if (data.result) {
            let jsonText = data.result.replace(/```json/gi, '').replace(/```/g, '').trim();
            try {
              const parsed = JSON.parse(jsonText);
              if (parsed.results) {
                Object.assign(updatedInsights, parsed.results);
              }
            } catch (e) {
              console.error("Failed to parse batch JSON:", e);
            }
          }
        } else if (res.status === 429) {
          console.warn("Rate limit hit during batching, skipping remaining chunks.");
          break; // Stop fetching if we hit a rate limit
        }
      }
      setBatchedInsights(updatedInsights);
    } catch (err) {
      console.error("Batch fetch error:", err);
    } finally {
      setIsBatchFetching(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#F7F7F4] text-[#102A43] font-sans antialiased">
      {/* 1. Left Vertical Icon Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={(tab) => setActiveTab(tab)}
        criticalAlertCount={activeAlertCount}
      />

      {/* 2. Outer Canvas Shell */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#F7F7F4] p-2 sm:p-4 md:p-6 transition-all">
        {/* Main Inner Scroll Container */}
        <main
          id="chronos-main-container"
          className="flex-1 w-full bg-[#F7F7F4] rounded-2xl sm:rounded-3xl border border-[#DCE4E7] shadow-2xs flex flex-col p-4 sm:p-6 lg:p-7 overflow-y-auto"
        >
          {/* Top Global Header with live telemetry */}
          <Header
            lastUpdated={mounted && systemStatus ? new Date().toLocaleTimeString() : 'Waiting...'}
            connected={connected}
            systemStatus={systemStatus}
            patientCount={patientCount}
            onOpenShortcuts={() => {}}
            searchQuery={searchQuery}
            onSearchChange={(q) => setSearchQuery(q)}
            onRefreshInsights={handleRefreshAllInsights}
            isBatchFetching={isBatchFetching}
          />

          {/* Dynamic Tab Views */}
          <div className="p-4 md:p-6 lg:p-8 w-full max-w-[1600px] mx-auto space-y-6 md:space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {activeTab === 'ward' && !selectedPatientId && (
              <WardDashboard
                patients={patients}
                sparklines={sparklines}
                batchedInsights={batchedInsights}
                onSelectPatient={(pid) => setSelectedPatientId(pid)}
              />
            )}

            {activeTab === 'ward' && selectedPatientId && patients[selectedPatientId] && (
              <SplitScreenView
                patient={patients[selectedPatientId]}
                onBack={() => setSelectedPatientId(null)}
              />
            )}

            {activeTab === 'alerts' && (
              <AlertFeed
                alerts={alerts}
                onAcknowledgeAlert={acknowledgeAlert}
              />
            )}
            
            {activeTab === 'analytics' && (
              <AnalyticsDashboard
                patients={patients}
                alerts={alerts}
              />
            )}
            
            {activeTab === 'settings' && (
              <div className="flex flex-col items-center justify-center p-12 bg-white rounded-2xl border border-slate-200 h-64">
                <h3 className="text-lg font-bold text-[#102A43]">System Settings</h3>
                <p className="text-sm text-[#52677D]">Configuration modules are restricted to Admins.</p>
              </div>
            )}
          </div>

          {/* Bottom Footer Reference */}
          <footer className="mt-8 pt-4 border-t border-[#DCE4E7] flex flex-col sm:flex-row items-center justify-between text-xs text-[#52677D] font-sans gap-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#4FA8B8]" />
              <span className="font-bold text-[#102A43]">CHRONOS</span>
              <span>• ICU Early Warning System & Patient Telemetry</span>
            </div>
            <div className="flex items-center gap-3 text-[11px]">
              <span className={connected ? "text-[#159A73] font-bold" : "text-[#F46B52] font-bold"}>
                {connected ? "WebSocket Connected" : "Connection Offline"}
              </span>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
