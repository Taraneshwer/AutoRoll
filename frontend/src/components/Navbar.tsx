import React, { useState, useEffect } from 'react';
import { Activity, Bell, Wifi, WifiOff, Lock, AlertTriangle } from 'lucide-react';
import { useWebSocketStatus, useDashboardMetrics } from '../hooks/useAutoRollData';

interface NavbarProps {
  activeTabTitle: string;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTabTitle }) => {
  const [timeStr, setTimeStr] = useState<string>('');
  const { status: wsStatus, isConnected } = useWebSocketStatus();
  const { metrics, error: metricsError } = useDashboardMetrics();

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeStr(new Date().toLocaleTimeString());
    }, 1000);
    setTimeStr(new Date().toLocaleTimeString());
    return () => clearInterval(timer);
  }, []);

  const getWsBadge = () => {
    switch (wsStatus) {
      case 'CONNECTED':
        return (
          <div className="flex items-center gap-1.5 text-xs text-emerald-700 font-semibold bg-emerald-50/90 px-2.5 py-1 rounded-md border border-emerald-200/80 shadow-2xs">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="hidden sm:inline">Telemetry Live</span>
          </div>
        );
      case 'CONNECTING':
      case 'RECONNECTING':
        return (
          <div className="flex items-center gap-1.5 text-xs text-amber-700 font-semibold bg-amber-50/90 px-2.5 py-1 rounded-md border border-amber-200/80 shadow-2xs">
            <Wifi className="w-3.5 h-3.5 text-amber-600 animate-pulse" />
            <span className="hidden sm:inline">Telemetry {wsStatus.toLowerCase()}...</span>
          </div>
        );
      case 'ERROR':
      case 'DISCONNECTED':
      default:
        return (
          <div className="flex items-center gap-1.5 text-xs text-rose-600 font-semibold bg-rose-50/90 px-2.5 py-1 rounded-md border border-rose-200/80 shadow-2xs">
            <WifiOff className="w-3.5 h-3.5 text-rose-600" />
            <span className="hidden sm:inline">Telemetry Offline</span>
          </div>
        );
    }
  };

  const getClusterHealthBadge = () => {
    if (metricsError) {
      return (
        <span className="px-2.5 py-1 text-xs font-bold rounded-md bg-rose-50 text-rose-600 border border-rose-200 flex items-center gap-1.5 shadow-2xs">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
          Backend Disconnected
        </span>
      );
    }

    if (!metrics) {
      return (
        <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-slate-100 text-slate-500 border border-slate-200 flex items-center gap-1.5">
          Connecting...
        </span>
      );
    }

    return (
      <span className="px-2.5 py-1 text-xs font-bold rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200/80 flex items-center gap-1.5 shadow-2xs">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
        Backend Online ({metrics.workersOnline}/{metrics.workersTotal} Nodes)
      </span>
    );
  };

  return (
    <header className="h-16 backdrop-blur-md bg-white/90 border-b border-slate-200/80 px-6 md:px-8 flex items-center justify-between sticky top-0 z-20 shadow-2xs shrink-0">
      {/* Left Area: Title + Status Badge */}
      <div className="flex items-center gap-4">
        <h2 className="text-[20px] font-extrabold text-slate-900 capitalize tracking-tight leading-none">
          {activeTabTitle}
        </h2>
        {getClusterHealthBadge()}
      </div>

      {/* Center/Right Area: Time, Privacy, Telemetry, Notifications */}
      <div className="flex items-center gap-4">
        {/* System Time */}
        <div className="text-xs font-mono text-slate-700 bg-slate-50/80 px-3 py-1.5 rounded-lg border border-slate-200/80 flex items-center gap-2 shadow-2xs">
          <Activity className="w-3.5 h-3.5 text-[#007FFF]" />
          <span>{timeStr || '--:--:--'}</span>
        </div>

        {/* Privacy Lock Badge */}
        <div className="hidden xl:flex items-center gap-1.5 text-xs text-slate-600 bg-slate-50/80 px-3 py-1.5 rounded-lg border border-slate-200/80 shadow-2xs">
          <Lock className="w-3.5 h-3.5 text-[#007FFF]" />
          <span>Privacy Preserved (No Image Storage)</span>
        </div>

        {/* Live WS Connection Indicator */}
        {getWsBadge()}

        {/* Notification Bell */}
        <button className="relative p-2 text-slate-500 hover:text-slate-800 rounded-lg hover:bg-slate-100 transition-colors">
          <Bell className="w-4 h-4" />
          {isConnected && <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-[#007FFF] rounded-full"></span>}
        </button>
      </div>
    </header>
  );
};
