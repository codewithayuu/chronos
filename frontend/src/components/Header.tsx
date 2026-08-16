import React from 'react';
import {
  Search,
  Keyboard,
  HeartPulse,
  Users,
} from 'lucide-react';
import { SystemStatus } from '../types';

interface HeaderProps {
  lastUpdated: string;
  connected: boolean;
  systemStatus: SystemStatus | null;
  patientCount: number;
  onOpenShortcuts: () => void;
  onOpenCommandPalette?: () => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onRefreshInsights?: () => void;
  isBatchFetching?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  lastUpdated,
  connected,
  systemStatus,
  patientCount,
  onOpenShortcuts,
  onOpenCommandPalette,
  searchQuery,
  onSearchChange,
  onRefreshInsights,
  isBatchFetching,
}) => {
  return (
    <header
      id="chronos-main-header"
      className="flex flex-col xl:flex-row items-stretch xl:items-center justify-between gap-4 mb-6 pb-5 border-b border-[#DCE4E7]"
    >
      {/* 1. Left: Product Title, Version & Status Indicators */}
      <div className="flex flex-col lg:flex-row lg:items-center gap-4 xl:gap-6">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-[28px] font-extrabold tracking-tight text-[#102A43] font-sans">
              CHRONOS
            </h1>
            <span className="px-2.5 py-0.5 rounded-lg bg-[#DCEEF3] text-[#4FA8B8] border border-[#4FA8B8]/20 text-[11px] font-sans font-bold tracking-wider">
              ICU v3.4
            </span>
          </div>
          <p className="text-xs font-bold text-[#52677D] tracking-wider uppercase mt-0.5 font-sans">
            EARLY WARNING SYSTEM
          </p>
        </div>

        {/* Live System Status Badges */}
        <div className="flex flex-nowrap items-center gap-2 pt-1 lg:pt-0 lg:pl-6 lg:border-l lg:border-[#DCE4E7]">
          {/* Status 1: System Connected */}
          <div className={`h-8.5 flex items-center gap-2 px-3 rounded-xl bg-white border border-[#DCE4E7] ${connected ? 'text-[#159A73]' : 'text-[#F46B52]'} text-xs font-semibold font-sans shadow-2xs`}>
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-[#159A73]' : 'bg-[#F46B52]'} animate-pulse`} />
            <span>{connected ? 'System Connected' : 'Disconnected'}</span>
          </div>

          {/* Status 2: Active Patients */}
          <div className="h-8.5 flex items-center gap-1.5 px-3 rounded-xl bg-white border border-[#DCE4E7] text-[#102A43] text-xs font-semibold font-sans shadow-2xs">
            <Users className="w-3.5 h-3.5 text-[#4FA8B8]" />
            <span><strong>{patientCount}</strong> Patients</span>
          </div>

          {/* Status 3: ML Engine status */}
          {systemStatus && (
            <div className="h-8.5 hidden sm:flex items-center gap-1.5 px-3 rounded-xl bg-white border border-[#DCE4E7] text-[#102A43] text-xs font-semibold font-sans shadow-2xs">
              <span className="text-[#52677D]">ML Engine:</span>
              <strong className="text-[#102A43]">{systemStatus.progress != null ? `Replay ${systemStatus.progress}%` : "Active"}</strong>
            </div>
          )}

          {/* Status 4: Last Update */}
          <div className="h-8.5 hidden md:flex items-center gap-1.5 px-3 rounded-xl bg-white border border-[#DCE4E7] text-xs text-[#52677D] font-sans shadow-2xs">
            <span>Last sync:</span>
            <strong className="text-[#102A43] font-mono">{lastUpdated || 'Waiting...'}</strong>
          </div>
        </div>
      </div>

      {/* 2. Right: Search & Control Tools */}
      <div className="flex flex-wrap items-center gap-2.5 justify-between xl:justify-end">
        {/* Universal Search Bar */}
        <div
          onClick={() => onOpenCommandPalette && onOpenCommandPalette()}
          className="relative w-full sm:w-72 md:w-80 cursor-pointer"
        >
          <Search className="w-4 h-4 text-[#52677D] absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search patients, alerts..."
            className="w-full bg-white hover:bg-[#F7F7F4] text-[#102A43] text-xs font-sans font-medium pl-9 pr-14 py-2 rounded-xl border border-[#DCE4E7] focus:outline-none focus:border-[#4FA8B8] focus:bg-white transition-all placeholder:text-[#52677D] shadow-2xs"
          />
          <div className="absolute right-2.5 top-1/2 -translate-y-1/2 flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 rounded bg-[#F7F7F4] border border-[#DCE4E7] text-[10px] font-mono text-[#52677D] shadow-2xs">
              ⌘K
            </kbd>
          </div>
        </div>

        {/* Shortcuts Button */}
        <button
          onClick={onOpenShortcuts}
          className="h-9 w-9 rounded-xl bg-white hover:bg-[#F7F7F4] border border-[#DCE4E7] flex items-center justify-center text-[#52677D] hover:text-[#102A43] transition-colors cursor-pointer shadow-2xs"
          title="Keyboard Shortcuts (Press ?)"
        >
          <Keyboard className="w-4 h-4" />
        </button>

        {/* Refresh All Insights Button */}
        {onRefreshInsights && (
          <button
            onClick={onRefreshInsights}
            disabled={isBatchFetching}
            className="h-9 flex items-center gap-1.5 px-3 rounded-xl bg-white hover:bg-[#F7F7F4] border border-[#4FA8B8]/30 text-[#4FA8B8] text-xs font-bold shadow-xs cursor-pointer transition-all disabled:opacity-50"
            title="Refresh AI Insights for all patients"
          >
            {isBatchFetching ? (
              <HeartPulse className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <HeartPulse className="w-3.5 h-3.5" />
            )}
            <span>{isBatchFetching ? 'Analyzing...' : 'Refresh AI'}</span>
          </button>
        )}

        {/* Live Engine Primary Action Button */}
        <div className="h-9 flex items-center gap-1.5 px-4 rounded-xl bg-[#4FA8B8] hover:bg-[#3f98a8] text-white text-xs font-bold shadow-xs select-none border border-[#4FA8B8] cursor-pointer transition-all">
          <HeartPulse className="w-3.5 h-3.5 text-white" />
          <span>Live Monitor</span>
        </div>
      </div>
    </header>
  );
};
