import React, { useState, useEffect } from 'react';
import { Camera as CameraIcon, ShieldCheck, Cpu, User, AlertTriangle, Play, Square } from 'lucide-react';

interface TelemetryFace {
  bbox: [number, number, number, number];
  detection_confidence: number;
  is_live: boolean;
  liveness_score: number;
  student_id: string | null;
  similarity: number;
  decision: string;
}

interface TelemetryData {
  timestamp: number;
  pipeline_fps: number;
  camera_fps: number;
  total_latency_ms: number;
  capture_latency_ms: number;
  detection_latency_ms: number;
  alignment_latency_ms: number;
  liveness_latency_ms: number;
  recognition_latency_ms: number;
  matching_latency_ms: number;
  face_count: number;
  faces: TelemetryFace[];
  gpu_name: string;
  vram_used_mb: number;
  active_model_id: string;
  recognition_threshold: number;
}

export const LiveMonitoringPage: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState<boolean>(true);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;

    const connectWS = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/monitoring`;

      ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.data) {
            setTelemetry(payload.data);
          } else if (payload.pipeline_fps !== undefined) {
            setTelemetry(payload);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket telemetry:', err);
        }
      };

      ws.onerror = () => {
        setCameraError('WebSocket telemetry connection error.');
      };
    };

    connectWS();
    return () => {
      if (ws) ws.close();
    };
  }, []);

  const handleStartCamera = async () => {
    try {
      setCameraError(null);
      await fetch('/api/v1/camera/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_type: 'local', camera_index: 0 }),
      });
      setIsStreaming(true);
    } catch (err: any) {
      setCameraError(err.message || 'Failed to start camera.');
    }
  };

  const handleStopCamera = async () => {
    try {
      await fetch('/api/v1/camera/stop', { method: 'POST' });
      setIsStreaming(false);
    } catch (err: any) {
      console.error(err);
    }
  };

  return (
    <div className="animate-fade-in max-w-full">
      {/* Page Header Area */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
        <div>
          <h2 className="text-[24px] font-bold text-[#111827] tracking-tight leading-tight">
            Live Monitoring
          </h2>
          <p className="text-[13px] text-[#64748B] mt-1">
            Real-time webcam stream with SCRFD detection, MiniFASNet anti-spoofing, and ArcFace template matching
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {isStreaming ? (
            <button
              onClick={handleStopCamera}
              className="px-3.5 py-1.5 bg-rose-50 text-rose-600 border border-rose-200 rounded-lg text-xs font-semibold flex items-center gap-2 hover:bg-rose-100 transition-colors shadow-xs"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              Stop Stream
            </button>
          ) : (
            <button
              onClick={handleStartCamera}
              className="px-3.5 py-1.5 bg-[#EAF5FF] text-[#007FFF] border border-[#007FFF]/20 rounded-lg text-xs font-semibold flex items-center gap-2 hover:bg-[#D5EAFF] transition-colors shadow-xs"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Start Stream
            </button>
          )}
          <span className="flex items-center gap-2 px-3 py-1.5 bg-white border border-[#E2E8F0] rounded-lg text-xs font-semibold text-[#111827]">
            <span className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'}`}></span>
            {isStreaming ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>

      {/* Compact Alert Banner if Error */}
      {cameraError && (
        <div className="w-fit max-w-full h-[36px] px-3 py-1.5 my-3 rounded-lg bg-[#FEF2F2] border border-[#FECACA] text-xs font-medium text-[#B91C1C] flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-[#DC2626]" />
          <span>{cameraError}</span>
        </div>
      )}

      {/* Main Monitoring Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Video Stream Feed */}
        <div className="lg:col-span-2 bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-xs flex flex-col">
          <div className="p-3.5 bg-slate-950/90 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping"></span>
              <span className="text-xs font-bold text-slate-200 font-mono">Laptop / USB Webcam (Index 0)</span>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
              <span>FPS: <strong className="text-emerald-400">{telemetry?.pipeline_fps || 0}</strong></span>
              <span>Latency: <strong className="text-[#007FFF]">{telemetry?.total_latency_ms || 0} ms</strong></span>
            </div>
          </div>

          <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
            {isStreaming ? (
              <img
                src="/api/v1/camera/mjpeg"
                alt="Real-Time Webcam Feed"
                className="w-full h-full object-contain"
                onError={() => setCameraError('MJPEG stream connection interrupted.')}
              />
            ) : (
              <div className="text-center text-slate-500">
                <CameraIcon className="w-14 h-14 text-slate-700 mx-auto mb-2" />
                <p className="text-sm font-semibold text-slate-400">Camera Stream Stopped</p>
                <p className="text-xs text-slate-600 mt-1">Click 'Start Stream' above to resume monitoring</p>
              </div>
            )}
          </div>

          {/* Bottom Telemetry Metrics Strip */}
          <div className="p-3 bg-slate-950/90 border-t border-slate-800 grid grid-cols-4 gap-2 text-center text-xs font-mono">
            <div className="p-2 bg-slate-900/60 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">DETECTION</span>
              <span className="text-emerald-400 font-bold">{telemetry?.detection_latency_ms || 0} ms</span>
            </div>
            <div className="p-2 bg-slate-900/60 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">LIVENESS</span>
              <span className="text-purple-400 font-bold">{telemetry?.liveness_latency_ms || 0} ms</span>
            </div>
            <div className="p-2 bg-slate-900/60 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">ARCFACE REC</span>
              <span className="text-sky-400 font-bold">{telemetry?.recognition_latency_ms || 0} ms</span>
            </div>
            <div className="p-2 bg-slate-900/60 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">MATCHING</span>
              <span className="text-amber-400 font-bold">{telemetry?.matching_latency_ms || 0} ms</span>
            </div>
          </div>
        </div>

        {/* Right Col: Telemetry & Live Recognized Faces Sidebar */}
        <div className="space-y-6">
          {/* System Hardware & Model Badge */}
          <div className="bg-white p-5 rounded-xl border border-[#E2E8F0] shadow-xs space-y-3">
            <h3 className="text-sm font-bold text-[#111827] flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#007FFF]" />
              Active Hardware & ML Model
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-100">
                <span className="text-[#64748B]">Active Model:</span>
                <span className="font-bold text-[#007FFF] font-mono">{telemetry?.active_model_id || 'autoroll_v1'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-100">
                <span className="text-[#64748B]">Model Threshold:</span>
                <span className="font-bold text-[#111827] font-mono">{telemetry?.recognition_threshold || 0.0540}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-100">
                <span className="text-[#64748B]">GPU Device:</span>
                <span className="font-bold text-[#111827] font-mono truncate">{telemetry?.gpu_name || 'NVIDIA RTX 5060'}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-[#64748B]">VRAM Memory:</span>
                <span className="font-bold text-[#111827] font-mono">{telemetry?.vram_used_mb || 0} MB</span>
              </div>
            </div>
          </div>

          {/* Recognized Identities Panel */}
          <div className="bg-white p-5 rounded-xl border border-[#E2E8F0] shadow-xs space-y-4">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <h3 className="text-sm font-bold text-[#111827] flex items-center gap-2">
                <User className="w-4 h-4 text-[#007FFF]" />
                Detected Faces ({telemetry?.face_count || 0})
              </h3>
              <span className="px-2 py-0.5 bg-[#EAF5FF] text-[#007FFF] rounded-md text-[10px] font-bold">
                Live Analysis
              </span>
            </div>

            {(!telemetry?.faces || telemetry.faces.length === 0) ? (
              <div className="py-8 text-center text-[#64748B] text-xs font-medium border border-dashed border-[#E2E8F0] rounded-lg bg-[#F8FAFC]">
                No faces currently detected in view
              </div>
            ) : (
              <div className="space-y-3 max-h-[360px] overflow-y-auto">
                {telemetry.faces.map((f, idx) => {
                  const isPresent = f.decision === 'PRESENT';
                  const isSpoof = f.decision.includes('SPOOF');
                  const badgeColor = isPresent
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : isSpoof
                    ? 'bg-rose-50 text-rose-700 border-rose-200'
                    : 'bg-amber-50 text-amber-700 border-amber-200';

                  return (
                    <div key={idx} className="p-3 bg-[#F8FAFC] rounded-lg border border-[#E2E8F0] space-y-1.5">
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-xs text-[#111827]">
                          {f.student_id || 'UNKNOWN'}
                        </span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold border ${badgeColor}`}>
                          {f.decision}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-[#64748B] pt-1">
                        <div>Similarity: <strong className="text-[#111827]">{(f.similarity * 100).toFixed(1)}%</strong></div>
                        <div>Liveness: <strong className="text-[#111827]">{(f.liveness_score * 100).toFixed(1)}%</strong></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
