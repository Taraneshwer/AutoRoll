import React from 'react';
import { Cpu, AlertTriangle, RefreshCw, ServerOff } from 'lucide-react';
import { useWorkers } from '../hooks/useAutoRollData';

export const WorkersPage: React.FC = () => {
  const { workers, loading, error, refresh } = useWorkers();

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">ML Worker Nodes Cluster</h2>
          <p className="text-sm text-slate-500 mt-1">Edge inference hardware telemetry, memory, and RTSP stream assignment status</p>
        </div>
        <button
          onClick={refresh}
          className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-[#EAF5FF] hover:text-[#007FFF] hover:border-[#007FFF]/30 text-xs font-semibold flex items-center gap-1.5 transition-all shadow-xs"
          title="Refresh Worker Cluster Telemetry"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh</span>
        </button>
      </div>

      {loading && workers.length === 0 ? (
        <div className="bg-white p-12 rounded-xl border border-slate-200 text-center text-slate-500 text-sm">
          Loading worker cluster telemetry...
        </div>
      ) : error ? (
        <div className="p-4 bg-[#FEF2F2] border border-[#FECACA] rounded-xl text-xs text-[#B91C1C] flex items-center justify-center gap-2">
          <AlertTriangle className="w-4 h-4 text-[#DC2626]" />
          <span>Unable to load telemetry: {error}</span>
        </div>
      ) : workers.length === 0 ? (
        <div className="bg-white p-12 rounded-xl text-center text-slate-500 text-sm flex flex-col items-center justify-center gap-3 border border-dashed border-slate-200">
          <ServerOff className="w-10 h-10 text-slate-400" />
          <p className="font-bold text-slate-800">No ML Worker Nodes Online</p>
          <p className="text-xs text-slate-500">
            Start a worker node instance to connect it to the central server control plane.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {workers.map((w) => (
            <div key={w.id} className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs space-y-5 relative">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-[#EAF5FF] border border-[#007FFF]/20 flex items-center justify-center text-[#007FFF]">
                    <Cpu className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-slate-900 text-base">{w.id}</h3>
                    <p className="text-[10px] font-mono text-slate-500">{w.modelVersion || 'Model Version: N/A'}</p>
                  </div>
                </div>

                <span className={`px-2.5 py-1 text-[10px] font-bold rounded-full ${
                  w.state === 'BUSY'
                    ? 'bg-amber-50 text-amber-700 border border-amber-200'
                    : w.state === 'READY'
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : 'bg-slate-100 text-slate-600 border border-slate-200'
                }`}>
                  {w.state}
                </span>
              </div>

              {/* Hardware Metrics Stack */}
              <div className="space-y-3 bg-slate-50 p-4 rounded-lg border border-slate-200 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-500">Status State:</span>
                  <span className="font-bold text-[#007FFF]">{w.state}</span>
                </div>

                {w.host && (
                  <div className="flex justify-between font-mono">
                    <span className="text-slate-500">Node Host:</span>
                    <span className="text-slate-900 font-semibold">{w.host}</span>
                  </div>
                )}

                <div className="flex justify-between">
                  <span className="text-slate-500">CPU Usage:</span>
                  <span className="font-mono text-slate-900 font-semibold">
                    {w.cpuPercent !== null && w.cpuPercent !== undefined ? `${w.cpuPercent}%` : 'N/A'}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-500">RAM Used:</span>
                  <span className="font-mono text-slate-900 font-semibold">
                    {w.ramUsedMb !== null && w.ramUsedMb !== undefined ? `${w.ramUsedMb} MB` : 'N/A'}
                    {w.ramPercent !== null && w.ramPercent !== undefined ? ` (${w.ramPercent}%)` : ''}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-500">GPU Device:</span>
                  <span className="font-semibold text-[#007FFF]">{w.gpuAvailable ? w.gpuName || 'GPU' : 'None (CPU Mode)'}</span>
                </div>

                {w.gpuAvailable && w.gpuUtilizationPercent !== null && w.gpuUtilizationPercent !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">GPU Utilization:</span>
                    <span className="font-mono text-[#007FFF] font-semibold">{w.gpuUtilizationPercent}%</span>
                  </div>
                )}

                <div className="flex justify-between">
                  <span className="text-slate-500">Active Cameras:</span>
                  <span className="font-bold text-[#007FFF]">{w.activeCamerasCount} Streams</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-500">Inference FPS:</span>
                  <span className="font-mono text-amber-700 font-semibold">
                    {w.fps !== null && w.fps !== undefined ? `${w.fps} FPS` : 'N/A'}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-500">Inference Latency:</span>
                  <span className="font-mono text-[#007A99] font-semibold">
                    {w.avgInferenceLatencyMs !== null && w.avgInferenceLatencyMs !== undefined ? `${w.avgInferenceLatencyMs} ms` : 'N/A'}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-slate-500">Queue Depth:</span>
                  <span className="font-mono text-purple-700 font-semibold">
                    {w.queueDepth !== undefined ? w.queueDepth : 0} / 2
                  </span>
                </div>

                {w.lastHeartbeat && (
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>Last Heartbeat:</span>
                    <span>
                      {typeof w.lastHeartbeat === 'number'
                        ? new Date(w.lastHeartbeat * 1000).toLocaleTimeString()
                        : String(w.lastHeartbeat)}
                    </span>
                  </div>
                )}

              </div>

              {/* Action Buttons */}
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-100">
                <button
                  onClick={async () => {
                    try {
                      await fetch(`/api/v1/workers/${w.id}/drain`, { method: 'POST' });
                      refresh();
                    } catch (e) {
                      console.error(e);
                    }
                  }}
                  className="px-3 py-1.5 bg-amber-50 hover:bg-amber-100 text-amber-800 rounded-lg text-xs font-bold transition-colors border border-amber-200"
                >
                  Drain Worker
                </button>
                <button
                  onClick={() => refresh()}
                  className="px-3 py-1.5 bg-[#E0F2F7] hover:bg-[#BBE3EE] text-[#007A99] rounded-lg text-xs font-bold transition-colors border border-[#007A99]/20"
                >
                  Reconnect / Sync
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

