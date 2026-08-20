import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { useDashboardMetrics } from '../hooks/useAutoRollData';

export const AnalyticsPage: React.FC = () => {
  const { metrics, loading, error } = useDashboardMetrics();

  const formatValue = (val: number | null | undefined, suffix: string = ''): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${val}${suffix}`;
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">Verification & Threat Analytics</h2>
        <p className="text-sm text-slate-500 mt-1">Real-time control plane telemetry and security event metrics</p>
      </div>

      {error && (
        <div className="p-4 bg-[#FEF2F2] border border-[#FECACA] rounded-xl text-xs text-[#B91C1C] flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-[#DC2626] shrink-0" />
          <span>Backend Disconnected: Unable to fetch live analytics telemetry.</span>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-2">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Attendance Records Today</span>
          <div className="text-3xl font-bold text-emerald-600">
            {loading ? '...' : formatValue(metrics?.attendanceToday)}
          </div>
          <p className="text-xs text-slate-500">Verified facial check-ins</p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-2">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Active Streams</span>
          <div className="text-3xl font-bold text-[#007FFF]">
            {loading ? '...' : formatValue(metrics?.camerasActive)}
          </div>
          <p className="text-xs text-slate-500">Active RTSP worker streams</p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-2">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Average Latency</span>
          <div className="text-3xl font-bold text-[#007FFF]">
            {loading ? '...' : formatValue(metrics?.avgLatencyMs, ' ms')}
          </div>
          <p className="text-xs text-slate-500">Inference pipeline latency</p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs space-y-2">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Spoof Attempts Blocked</span>
          <div className="text-3xl font-bold text-amber-600">
            {loading ? '...' : formatValue(metrics?.spoofAttempts)}
          </div>
          <p className="text-xs text-slate-500">Presentation attacks blocked</p>
        </div>
      </div>

      {/* Security Telemetry Status */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs space-y-4">
        <h3 className="text-lg font-bold text-slate-900">Presentation Attack Threats</h3>

        {metrics && metrics.spoofAttempts !== null && metrics.spoofAttempts !== undefined ? (
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 flex justify-between items-center text-xs">
            <span className="text-slate-700 font-semibold">Total Spoof Attempts Blocked:</span>
            <span className="font-bold text-[#DC2626] text-sm">{metrics.spoofAttempts}</span>
          </div>
        ) : (
          <div className="p-8 text-center text-xs text-slate-500 border border-dashed border-slate-200 rounded-lg">
            No active threat telemetry events reported by workers.
          </div>
        )}
      </div>
    </div>
  );
};
