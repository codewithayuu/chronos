import React, { useState } from 'react';
import { AlertData } from '../types';
import {
  AlertTriangle,
  Search,
  ShieldAlert,
  Flame,
  CheckCircle2,
  Sparkles,
} from 'lucide-react';

interface AlertFeedProps {
  alerts: AlertData[];
  onAcknowledgeAlert: (alertId: string) => void;
}

export const AlertFeed: React.FC<AlertFeedProps> = ({
  alerts,
  onAcknowledgeAlert,
}) => {
  const [filterUnacknowledged, setFilterUnacknowledged] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const filteredAlerts = alerts.filter((alert) => {
    if (filterUnacknowledged && alert.acknowledged) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return (
        alert.message.toLowerCase().includes(q) ||
        alert.patient_id.toLowerCase().includes(q) ||
        alert.severity.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return 'text-[#F46B52] bg-[#FFE0D6] border-[#F46B52]';
      case 'WARNING':
        return 'text-[#E9A52F] bg-[#F8E6A7] border-[#ebd488]';
      default:
        return 'text-[#4FA8B8] bg-[#DCEEF3] border-[#4FA8B8]';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return <Flame className="w-4 h-4 text-[#F46B52]" />;
      case 'WARNING':
        return <ShieldAlert className="w-4 h-4 text-[#E9A52F]" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-[#4FA8B8]" />;
    }
  };

  return (
    <div
      id="alert-feed-panel"
      className="p-5 sm:p-6 rounded-2xl bg-white border border-slate-200 flex flex-col h-full min-h-[640px] shadow-2xs"
    >
      {/* 1. Header & Quick Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 mb-4 border-b border-slate-200">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-lg font-bold text-[#102A43] tracking-tight font-sans">
              ICU ALERTS
            </h2>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-md bg-[#DCEEF3] text-[#4FA8B8] border border-[#4FA8B8]">
              {alerts.length} Total
            </span>
          </div>
          <p className="text-sm text-[#52677D] mt-0.5 font-sans">
            Early warning notifications for critical patients
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setFilterUnacknowledged(!filterUnacknowledged)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold transition-all border cursor-pointer min-h-[38px] ${
              filterUnacknowledged
                ? 'bg-[#FFE0D6] text-[#F46B52] border-[#F46B52] font-bold shadow-xs'
                : 'bg-white text-[#52677D] border-slate-200 hover:text-[#102A43] hover:border-slate-300'
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Unacknowledged</span>
          </button>

          <div className="relative">
            <Search className="w-4 h-4 text-[#52677D] absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search alerts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 pr-3 py-2 text-xs sm:text-sm rounded-xl bg-white border border-slate-200 text-[#102A43] placeholder-[#52677D] focus:outline-none focus:border-[#4FA8B8] font-sans w-36 sm:w-48 transition-colors shadow-2xs"
            />
          </div>
        </div>
      </div>

      {/* 2. Activity Timeline List */}
      <div className="flex-1 overflow-y-auto pr-1 space-y-3 dark-scrollbar">
        {filteredAlerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-[#52677D] text-xs sm:text-sm font-sans">
            <Sparkles className="w-8 h-8 mb-2 text-[#4FA8B8] opacity-40" />
            <span className="font-semibold text-[#102A43]">No active alerts</span>
            <span className="text-xs text-[#52677D] mt-0.5">All patients are stable.</span>
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            return (
              <div
                key={alert.alert_id}
                className={`p-4 sm:p-5 rounded-2xl transition-all duration-200 border relative ${
                  !alert.acknowledged
                    ? 'bg-[#FFFAF0] border border-[#F8E6A7]'
                    : 'bg-[#F8F8F5] border border-slate-200/90'
                }`}
              >
                {/* 1. Header Row */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <div className="p-1.5 rounded-lg bg-white border border-slate-200 shrink-0 shadow-2xs">
                      {getSeverityIcon(alert.severity)}
                    </div>

                    <span className={`text-xs font-bold px-2 py-0.5 rounded border ${getSeverityColor(alert.severity)} font-sans`}>
                      {alert.severity}
                    </span>

                    <span className="text-sm font-bold text-[#102A43] font-sans">
                      Patient: {alert.patient_id}
                    </span>

                    <span className="text-xs font-mono text-[#52677D] bg-white px-2 py-0.5 rounded border border-slate-200 shadow-2xs">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>

                {/* 2. ACTION: Primary Visual Focus */}
                <div className="text-base sm:text-[16px] font-bold text-[#102A43] mt-2.5 flex items-start gap-2 font-sans tracking-tight">
                  <AlertTriangle className="w-4 h-4 text-[#F56B52] shrink-0 mt-1" />
                  <span>{alert.message}</span>
                </div>

                {/* 3. AUTHORIZATION ROW */}
                <div className="mt-3 pt-3 border-t border-slate-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-sans">
                  <div className="flex items-center gap-2 flex-wrap">
                    {alert.acknowledged ? (
                      <span className="text-xs font-bold text-[#159A73] uppercase tracking-wider flex items-center gap-1">
                        <CheckCircle2 className="w-4 h-4" />
                        ACKNOWLEDGED BY {alert.acknowledged_by}
                      </span>
                    ) : (
                      <span className="text-xs font-bold text-[#F56B52] uppercase tracking-wider">
                        ACTION REQUIRED
                      </span>
                    )}
                  </div>

                  {!alert.acknowledged && (
                    <button
                      onClick={() => onAcknowledgeAlert(alert.alert_id)}
                      className="text-white bg-[#4FA8B8] hover:bg-[#3f98a8] px-3 py-1.5 rounded-lg font-bold text-xs flex items-center gap-1 transition-all cursor-pointer"
                    >
                      <span>Acknowledge</span>
                    </button>
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
