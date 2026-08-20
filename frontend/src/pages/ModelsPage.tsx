import React, { useEffect, useState } from 'react';
import { BrainCircuit, ShieldCheck, AlertTriangle, Cpu, Activity, RefreshCw } from 'lucide-react';

interface MLStatusData {
  status: string;
  active_model_id: string;
  model_version: string;
  recognition_threshold: number;
  embedding_dimension: number;
  device: string;
  providers: string[];
  liveness_model: string;
  liveness_threshold: number;
  detector_model: string;
  temporal_required_observations: number;
  temporal_confirmation_window_ms: number;
}

export const ModelsPage: React.FC = () => {
  const [mlStatus, setMlStatus] = useState<MLStatusData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/v1/ml/status');
      if (!res.ok) throw new Error('Failed to fetch ML status');
      const data = await res.json();
      setMlStatus(data);
    } catch (err: any) {
      setError(err.message || 'Error connecting to backend ML status service.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">System & ML Model Status</h2>
          <p className="text-sm text-slate-500 mt-1">
            ArcFace feature extractor, MiniFASNet anti-spoofing, SCRFD detector, and hardware runtime telemetry
          </p>
        </div>
        <button
          onClick={fetchStatus}
          className="px-3.5 py-2 bg-white border border-slate-200 hover:bg-[#EAF5FF] hover:text-[#007FFF] rounded-xl text-xs font-bold flex items-center gap-2 shadow-xs transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="bg-white p-12 rounded-2xl border border-slate-200 text-center text-slate-500 text-sm font-semibold">
          Fetching ML runtime status...
        </div>
      ) : mlStatus && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Card 1: Recognition Model Status */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-[#EAF5FF] border border-[#007FFF]/20 flex items-center justify-center text-[#007FFF]">
                  <BrainCircuit className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-base">ArcFace Feature Extractor</h3>
                  <p className="text-xs text-slate-500">Recognition Model Subsystem</p>
                </div>
              </div>
              <span className="px-2.5 py-1 text-[10px] font-extrabold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                READY
              </span>
            </div>

            <div className="space-y-2 bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-500">Active Model ID:</span>
                <span className="text-[#007FFF] font-bold">{mlStatus.active_model_id}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-500">Model Version:</span>
                <span className="text-slate-900 font-semibold">{mlStatus.model_version}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-500">Validated Threshold:</span>
                <span className="text-emerald-700 font-bold">{mlStatus.recognition_threshold}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Embedding Dimension:</span>
                <span className="text-purple-700 font-bold">{mlStatus.embedding_dimension}-d normalized</span>
              </div>
            </div>
          </div>

          {/* Card 2: Hardware Execution & Pipeline Parameters */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-[#EAF5FF] border border-[#007FFF]/20 flex items-center justify-center text-[#007FFF]">
                  <Cpu className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-base">Hardware Execution & Liveness</h3>
                  <p className="text-xs text-slate-500">Anti-Spoofing & Detector Settings</p>
                </div>
              </div>
              <span className="px-2.5 py-1 text-[10px] font-extrabold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                {mlStatus.device.toUpperCase()}
              </span>
            </div>

            <div className="space-y-2 bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-500">Detector Model:</span>
                <span className="text-slate-900 font-semibold">{mlStatus.detector_model}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-500">Liveness Model:</span>
                <span className="text-slate-900 font-semibold">{mlStatus.liveness_model}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-200">
                <span className="text-slate-500">Liveness Threshold:</span>
                <span className="text-emerald-700 font-bold">{mlStatus.liveness_threshold}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Temporal Confirmation:</span>
                <span className="text-[#007FFF] font-bold">{mlStatus.temporal_required_observations} obs / {mlStatus.temporal_confirmation_window_ms} ms</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
