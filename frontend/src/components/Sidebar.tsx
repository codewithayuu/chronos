import React from 'react';
import {
  Activity,
  Users,
  Bell,
  Settings,
  HeartPulse,
} from 'lucide-react';

export type NavTabId = 'ward' | 'alerts' | 'analytics' | 'settings';

interface SidebarProps {
  activeTab: NavTabId;
  onSelectTab: (tab: NavTabId) => void;
  criticalAlertCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  criticalAlertCount,
}) => {
  const navItems: Array<{
    id: NavTabId;
    label: string;
    description: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: string | number;
    badgeColor?: string;
  }> = [
    {
      id: 'ward',
      label: 'Ward View',
      description: 'Active patients & real-time telemetry',
      icon: Users,
    },
    {
      id: 'alerts',
      label: 'Active Alerts',
      description: 'Critical warnings and system alerts',
      icon: Bell,
      badge: criticalAlertCount > 0 ? `${criticalAlertCount}` : undefined,
      badgeColor: 'bg-[#F46B52]',
    },
    {
      id: 'analytics',
      label: 'Analytics',
      description: 'Historical data & composite entropy',
      icon: Activity,
    },
    {
      id: 'settings',
      label: 'Settings',
      description: 'System configuration',
      icon: Settings,
    },
  ];

  return (
    <>
      {/* Desktop & Tablet Sidebar Rail */}
      <aside
        id="chronos-left-sidebar"
        className="hidden md:flex w-16 lg:w-20 bg-[#F7F7F4] flex-col items-center py-5 border-r border-[#DCE4E7] select-none shrink-0 transition-all z-20"
      >
        {/* Top Logo */}
        <div
          className="w-11 h-11 rounded-2xl bg-white border border-[#DCE4E7] p-0.5 flex items-center justify-center shadow-2xs mb-6 cursor-pointer hover:border-[#4FA8B8]/50 hover:shadow-xs transition-all group relative"
          onClick={() => onSelectTab('ward')}
          title="CHRONOS Dashboard"
        >
          <div className="w-full h-full bg-white rounded-[13px] flex items-center justify-center text-[#102A43]">
            <HeartPulse className="w-5 h-5 text-[#4FA8B8] group-hover:scale-110 transition-transform" />
          </div>
          {/* Rich Tooltip */}
          <div className="absolute left-full ml-3 px-3 py-2 bg-white text-[#102A43] text-xs font-semibold rounded-xl whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 shadow-xl border border-[#DCE4E7]">
            <div className="text-xs font-bold text-[#102A43] font-sans">CHRONOS</div>
            <div className="text-[11px] text-[#52677D] font-normal font-sans">ICU Early Warning System</div>
          </div>
        </div>

        {/* Nav Icons List */}
        <nav className="flex-1 flex flex-col items-center gap-2.5 w-full px-2 sm:px-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                id={`nav-tab-${item.id}`}
                onClick={() => onSelectTab(item.id)}
                className={`relative group w-11 h-11 lg:w-12 lg:h-12 rounded-xl flex items-center justify-center transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-[#DCEEF3] text-[#4FA8B8] border border-[#4FA8B8]/30 shadow-2xs'
                    : 'text-[#52677D] hover:text-[#102A43] hover:bg-white border border-transparent hover:border-slate-200'
                }`}
              >
                <Icon
                  className={`w-5 h-5 transition-transform ${
                    isActive ? 'text-[#4FA8B8]' : 'text-[#52677D] group-hover:text-[#102A43]'
                  }`}
                />

                {/* Subtle Left Vertical Indicator */}
                {isActive && (
                  <span className="absolute -left-2 lg:-left-3 w-1 h-5 rounded-r-full bg-[#4FA8B8]" />
                )}

                {/* Optional Notification Dot */}
                {item.badge && !isActive && (
                  <span
                    className={`absolute top-2 right-2 w-2 h-2 rounded-full ${
                      item.badgeColor || 'bg-[#D49B1D]'
                    } animate-pulse`}
                  />
                )}

                {/* Rich Hover Tooltip with Label & Description */}
                <div className="absolute left-full ml-3 px-3 py-2 bg-white text-left rounded-xl whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 shadow-xl border border-[#DCE4E7]">
                  <div className="text-xs font-bold text-[#102A43] font-sans">{item.label}</div>
                  <div className="text-[11px] text-[#52677D] font-normal font-sans">{item.description}</div>
                </div>
              </button>
            );
          })}
        </nav>

        {/* Bottom Utility Controls (Alerts & Live Status) */}
        <div className="mt-auto flex flex-col items-center gap-3 pt-4 border-t border-[#DCE4E7] w-full px-2">
          <button
            onClick={() => onSelectTab('alerts')}
            className="relative group w-10 h-10 rounded-xl flex items-center justify-center text-[#52677D] hover:text-[#102A43] hover:bg-white transition-colors cursor-pointer border border-transparent hover:border-slate-200"
            title="Alerts"
          >
            <Bell className="w-4 h-4" />
            {criticalAlertCount > 0 && (
              <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-[#F46B52]" />
            )}
            <div className="absolute left-full ml-3 px-3 py-1.5 bg-white text-[#102A43] text-xs font-normal rounded-xl whitespace-nowrap opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 shadow-xl border border-[#DCE4E7]">
              Alerts ({criticalAlertCount > 0 ? `${criticalAlertCount} Active` : 'Nominal'})
            </div>
          </button>

          {/* System Pulse */}
          <div className="flex items-center justify-center p-1.5" title="System monitoring nominal">
            <div className={`w-2.5 h-2.5 rounded-full ${criticalAlertCount > 0 ? 'bg-[#F46B52] ring-[#F46B52]/20' : 'bg-[#159A73] ring-[#159A73]/20'} ring-4 shadow-xs animate-pulse`} />
          </div>
        </div>
      </aside>

      {/* Mobile Bottom Navigation Bar (<768px) */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-[#F7F7F4] border-t border-[#DCE4E7] z-40 flex items-center justify-around px-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex flex-col items-center justify-center w-12 h-12 rounded-xl transition-colors cursor-pointer relative ${
                isActive ? 'text-[#4FA8B8] bg-[#DCEEF3]' : 'text-[#52677D]'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="text-[9px] font-sans mt-0.5 font-medium truncate max-w-[48px]">
                {item.label.split(' ')[0]}
              </span>
              {item.badge && (
                <span className={`absolute top-1 right-2 w-2 h-2 rounded-full ${item.badgeColor || 'bg-[#F46B52]'}`} />
              )}
            </button>
          );
        })}
      </div>
    </>
  );
};
