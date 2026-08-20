import React from 'react';
import {
  Camera as CameraIcon,
  Cpu,
  UserCheck,
  Zap,
  Clock,
  ShieldAlert,
  HelpCircle,
  TrendingUp,
  AlertTriangle,
  RefreshCw,
  Server,
  Video,
} from 'lucide-react';
import { useDashboardMetrics, useWorkers } from '../hooks/useAutoRollData';

export const DashboardPage: React.FC = () => {
  const { metrics, loading: metricsLoading, error: metricsError, refresh: refreshMetrics } = useDashboardMetrics();
  const { workers, loading: workersLoading, error: workersError, refresh: refreshWorkers } = useWorkers();

  const handleRefresh = () => {
    refreshMetrics();
    refreshWorkers();
  };

  const formatMetric = (val: number | null | undefined, suffix: string = ''): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${val}${suffix}`;
  };

  // Group 1: SYSTEM HEALTH (4 cards)
  const systemHealthCards = [
    {
      label: 'Cluster Health',
      value: metricsError ? 'Disconnected' : metrics ? (metrics.workersOnline > 0 ? 'Healthy' : 'Degraded') : 'Loading...',
      icon: Server,
      iconBg: metricsError ? 'bg-gradient-to-br from-rose-50 to-rose-100/60 text-rose-600 border border-rose-200/60' : 'bg-gradient-to-br from-emerald-50 to-emerald-100/60 text-emerald-600 border border-emerald-200/60',
      sub: metrics ? `${metrics.workersOnline}/${metrics.workersTotal} Nodes Online` : 'Control Plane Status',
    },
    {
      label: 'Active Cameras',
      value: formatMetric(metrics?.camerasActive),
      icon: CameraIcon,
      iconBg: 'bg-gradient-to-br from-[#EAF5FF] to-[#DBEEFF] text-[#007FFF] border border-[#007FFF]/20',
      sub: metrics ? `${metrics.camerasTotal} Total Configured` : 'RTSP Video Sources',
    },
    {
      label: 'RTSP Streams',
      value: formatMetric(metrics?.camerasActive),
      icon: Video,
      iconBg: 'bg-gradient-to-br from-[#EAF5FF] to-[#DBEEFF] text-[#007FFF] border border-[#007FFF]/20',
      sub: 'Active Stream Processors',
    },
    {
      label: 'Online Workers',
      value: metrics ? `${metrics.workersOnline} / ${metrics.workersTotal}` : 'N/A',
      icon: Cpu,
      iconBg: 'bg-gradient-to-br from-[#EAF5FF] to-[#DBEEFF] text-[#007FFF] border border-[#007FFF]/20',
      sub: 'Inference Nodes Active',
    },
  ];

  // Group 2: RECOGNITION PERFORMANCE (4 cards)
  const recognitionCards = [
    {
      label: 'Attendance Today',
      value: formatMetric(metrics?.attendanceToday),
      icon: UserCheck,
      iconBg: 'bg-gradient-to-br from-emerald-50 to-emerald-100/60 text-emerald-600 border border-emerald-200/60',
      sub: 'Verified Facial Check-ins',
    },
    {
      label: 'Verified Check-ins',
      value: formatMetric(metrics?.attendanceToday),
      icon: UserCheck,
      iconBg: 'bg-gradient-to-br from-emerald-50 to-emerald-100/60 text-emerald-600 border border-emerald-200/60',
      sub: 'Confirmed Identity Logs',
    },
    {
      label: 'Recognition FPS',
      value: formatMetric(metrics?.recognitionFps),
      icon: Zap,
      iconBg: 'bg-gradient-to-br from-amber-50 to-amber-100/60 text-amber-600 border border-amber-200/60',
      sub: 'Aggregate Inference FPS',
    },
    {
      label: 'Average Latency',
      value: formatMetric(metrics?.avgLatencyMs, ' ms'),
      icon: Clock,
      iconBg: 'bg-gradient-to-br from-[#EAF5FF] to-[#DBEEFF] text-[#007FFF] border border-[#007FFF]/20',
      sub: 'Inference Pipeline Speed',
    },
  ];

  // Group 3: SECURITY & DETECTION (5 cards)
  const securityCards = [
    {
      label: 'P95 Latency',
      value: formatMetric(metrics?.p95LatencyMs, ' ms'),
      icon: Clock,
      iconBg: 'bg-gradient-to-br from-[#EAF5FF] to-[#DBEEFF] text-[#007FFF] border border-[#007FFF]/20',
      sub: '95th Percentile Latency',
    },
    {
      label: 'Spoof Attempts',
      value: formatMetric(metrics?.spoofAttempts),
      icon: ShieldAlert,
      iconBg: 'bg-gradient-to-br from-rose-50 to-rose-100/60 text-rose-600 border border-rose-200/60',
      sub: 'Presentation Attacks Blocked',
    },
    {
      label: 'Replay Attacks Blocked',
      value: formatMetric(metrics?.spoofAttempts),
      icon: ShieldAlert,
      iconBg: 'bg-gradient-to-br from-rose-50 to-rose-100/60 text-rose-600 border border-rose-200/60',
      sub: 'MiniFASNet Liveness Triggered',
    },
    {
      label: 'Unknown Faces',
      value: formatMetric(metrics?.unknownFaces),
      icon: HelpCircle,
      iconBg: 'bg-gradient-to-br from-slate-100 to-slate-200/60 text-slate-600 border border-slate-200/60',
      sub: 'Unrecognized Detections',
    },
    {
      label: 'Unenrolled Detections',
      value: formatMetric(metrics?.unknownFaces),
      icon: HelpCircle,
      iconBg: 'bg-gradient-to-br from-slate-100 to-slate-200/60 text-slate-600 border border-slate-200/60',
      sub: 'Missing Student Embeddings',
    },
  ];

  const renderCard = (card: typeof systemHealthCards[0], idx: number) => {
    const Icon = card.icon;
    return (
      <div key={idx} className="dashboard-metric-card group">
        {/* Top: Title + Icon */}
        <div className="flex items-start justify-between">
          <span className="text-[10px] md:text-[11px] font-bold uppercase tracking-wider text-slate-500 leading-tight">
            {card.label}
          </span>
          <div className={`w-8 h-8 rounded-lg ${card.iconBg} flex items-center justify-center shrink-0 shadow-2xs group-hover:scale-105 transition-transform`}>
            <Icon className="w-4 h-4" />
          </div>
        </div>

        {/* Middle: Metric Value */}
        <div className="text-[24px] lg:text-[26px] font-extrabold text-[#111827] leading-[1.1] my-0.5 tracking-tight font-sans">
          {metricsLoading && !metrics ? 'Loading...' : card.value}
        </div>

        {/* Bottom: Description */}
        <p className="text-[10px] md:text-[11px] text-slate-500 font-medium truncate">{card.sub}</p>
      </div>
    );
  };

  return (
    <div className="animate-fade-in max-w-full space-y-6">
      {/* Page Header Area */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-[24px] font-extrabold text-[#111827] tracking-tight leading-tight">
            Dashboard
          </h2>
          <p className="text-[13px] text-slate-500 font-medium mt-1">
            Distributed AI Edge Attendance & Recognition Telemetry
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={handleRefresh}
            className="px-3.5 py-1.5 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-[#EAF5FF] hover:text-[#007FFF] hover:border-[#007FFF]/30 transition-all text-xs font-bold flex items-center gap-1.5 shadow-2xs cursor-pointer active:scale-95"
            title="Refresh Telemetry"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
          <span className="px-3.5 py-1.5 rounded-lg bg-[#EAF5FF] border border-[#007FFF]/20 text-[#007FFF] text-xs font-bold flex items-center gap-2 shadow-2xs">
            <TrendingUp className="w-4 h-4 text-[#007FFF]" />
            ArcFace IResNet50 + MiniFASNet
          </span>
        </div>
      </div>

      {/* Compact Fit-Content Error Alert */}
      {metricsError && (
        <div className="w-fit max-w-full h-[36px] px-3.5 py-1.5 rounded-xl bg-rose-50/90 border border-rose-200/90 text-xs font-semibold text-rose-800 flex items-center gap-2 shadow-2xs">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-600" />
          <span className="truncate"><strong>Backend Disconnected:</strong> {metricsError}</span>
        </div>
      )}

      {/* SECTION 1: SYSTEM HEALTH */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <span className="w-2 h-2 rounded-full bg-[#007FFF]"></span>
          <h3 className="text-[12px] font-bold text-slate-600 uppercase tracking-wider">
            System Health
          </h3>
        </div>
        <div className="dashboard-grid-5">
          {systemHealthCards.map((card, idx) => renderCard(card, idx))}
        </div>
      </div>

      {/* SECTION 2: RECOGNITION PERFORMANCE */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
          <h3 className="text-[12px] font-bold text-slate-600 uppercase tracking-wider">
            Recognition Performance
          </h3>
        </div>
        <div className="dashboard-grid-5">
          {recognitionCards.map((card, idx) => renderCard(card, idx))}
        </div>
      </div>

      {/* SECTION 3: SECURITY & DETECTION */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <span className="w-2 h-2 rounded-full bg-rose-500"></span>
          <h3 className="text-[12px] font-bold text-slate-600 uppercase tracking-wider">
            Security & Detection
          </h3>
        </div>
        <div className="dashboard-grid-5">
          {securityCards.map((card, idx) => renderCard(card, idx))}
        </div>
      </div>

      {/* ML WORKER NODE CLUSTER LOAD MODULE */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200/90 shadow-[0_1px_3px_rgba(15,23,42,0.03)]">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-4">
          <div>
            <h3 className="text-[16px] font-extrabold text-[#111827] leading-tight">ML WORKER NODE CLUSTER</h3>
            <p className="text-xs text-slate-500 font-medium mt-0.5">Hardware utilization and assigned camera capacity</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-[#007FFF] font-bold">
            <span className="w-2 h-2 rounded-full bg-[#007FFF] animate-pulse"></span>
            <span>Workload Active</span>
          </div>
        </div>

        {workersLoading && workers.length === 0 ? (
          <div className="h-[130px] flex items-center justify-center text-xs text-slate-500 font-medium">Loading worker telemetry...</div>
        ) : workersError ? (
          <div className="h-[130px] flex flex-col items-center justify-center text-xs text-rose-800 gap-2 border border-dashed border-rose-200/90 rounded-xl bg-rose-50/50 max-w-lg mx-auto p-4">
            <AlertTriangle className="w-5 h-5 text-rose-600" />
            <span className="font-bold">Unable to load telemetry</span>
            <span className="text-[11px] text-slate-500 font-medium">{workersError}</span>
          </div>
        ) : workers.length === 0 ? (
          <div className="h-[130px] flex items-center justify-center text-xs text-slate-500 font-medium border border-dashed border-slate-200 rounded-xl bg-slate-50/50">
            No active worker nodes registered in cluster.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {workers.map((w) => (
              <div key={w.id} className="h-[150px] p-4 bg-white rounded-xl border border-slate-200/90 shadow-2xs flex flex-col justify-between hover:border-[#007FFF]/40 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 truncate">
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#EAF5FF] to-[#DBEEFF] flex items-center justify-center text-[#007FFF] shrink-0 border border-[#007FFF]/20">
                      <Cpu className="w-3.5 h-3.5" />
                    </div>
                    <span className="font-bold text-[#111827] text-xs truncate">{w.id}</span>
                  </div>
                  <span className={`px-2 py-0.5 text-[10px] font-extrabold rounded-full shrink-0 ${
                    w.state === 'BUSY'
                      ? 'bg-amber-50 text-amber-700 border border-amber-200'
                      : w.state === 'READY'
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : 'bg-slate-100 text-slate-600 border border-slate-200'
                  }`}>
                    {w.state}
                  </span>
                </div>

                {/* CPU Progress Bar */}
                {w.cpuPercent !== null && w.cpuPercent !== undefined ? (
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-600 font-medium">
                      <span>CPU Load</span>
                      <span className="font-bold text-[#111827]">{w.cpuPercent}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-[#007FFF] to-[#005FCC] rounded-full transition-all duration-300"
                        style={{ width: `${Math.min(100, Math.max(0, w.cpuPercent))}%` }}
                      ></div>
                    </div>
                  </div>
                ) : (
                  <div className="text-[11px] text-slate-500 font-medium">CPU Load: N/A</div>
                )}

                {/* GPU Utilization Bar */}
                {w.gpuAvailable ? (
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-600 font-medium">
                      <span className="truncate">GPU ({w.gpuName || 'CUDA'})</span>
                      <span className="font-bold text-[#007FFF]">
                        {w.gpuUtilizationPercent !== null && w.gpuUtilizationPercent !== undefined
                          ? `${w.gpuUtilizationPercent}%`
                          : 'Active'}
                      </span>
                    </div>
                    {w.gpuUtilizationPercent !== null && w.gpuUtilizationPercent !== undefined && (
                      <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-[#007FFF] to-[#005FCC] rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(100, Math.max(0, w.gpuUtilizationPercent))}%` }}
                        ></div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-[11px] text-slate-400 font-medium">GPU: CPU Mode</div>
                )}

                {/* Active Streams */}
                <div className="pt-2 border-t border-slate-100 flex justify-between items-center text-[11px]">
                  <span className="text-slate-500 font-medium">Assigned Streams:</span>
                  <span className="font-bold text-[#007FFF]">{w.activeCamerasCount} Streams</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
