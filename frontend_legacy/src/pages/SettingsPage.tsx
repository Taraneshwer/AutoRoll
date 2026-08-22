import React, { useState } from 'react';
import { Sliders, Shield, Save, CheckCircle2 } from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [similarityThreshold, setSimilarityThreshold] = useState(0.65);
  const [livenessThreshold, setLivenessThreshold] = useState(0.90);
  const [dedupWindow, setDedupWindow] = useState(300);
  const [saved, setSaved] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">System Configuration & Thresholds</h2>
        <p className="text-sm text-slate-500 mt-1">Configure global recognition decision parameters and deduplication windows</p>
      </div>

      {saved && (
        <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center gap-3 text-emerald-800 text-xs font-semibold">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>System threshold configuration saved.</span>
        </div>
      )}

      <form onSubmit={handleSave} className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs space-y-6">
        {/* Similarity Threshold Slider */}
        <div className="space-y-2">
          <div className="flex justify-between items-center text-sm font-bold text-slate-900">
            <label className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[#007A99]" />
              <span>Cosine Similarity Threshold</span>
            </label>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-[10px] font-mono">Server-Controlled</span>
              <span className="font-mono text-[#007A99]">{similarityThreshold}</span>
            </div>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.20"
            step="0.001"
            value={similarityThreshold}
            onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-[#007A99]"
          />
          <p className="text-xs text-slate-500">Minimum Cosine similarity score required to accept student recognition (Default: 0.0540 for autoroll_v1).</p>
        </div>

        {/* Liveness Threshold Slider */}
        <div className="space-y-2 pt-4 border-t border-slate-100">
          <div className="flex justify-between items-center text-sm font-bold text-slate-900">
            <label className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-[#007A99]" />
              <span>Anti-Spoof Liveness Threshold</span>
            </label>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-[10px] font-mono">Server-Controlled</span>
              <span className="font-mono text-[#007A99]">{livenessThreshold}</span>
            </div>
          </div>
          <input
            type="range"
            min="0.70"
            max="0.99"
            step="0.01"
            value={livenessThreshold}
            onChange={(e) => setLivenessThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-[#007A99]"
          />
          <p className="text-xs text-slate-500">Minimum MiniFASNet probability required to classify a face sample as REAL.</p>
        </div>

        {/* Deduplication Window */}
        <div className="space-y-2 pt-4 border-t border-slate-100">
          <div className="flex justify-between items-center text-sm font-bold text-slate-900">
            <label className="block text-sm font-bold text-slate-900">
              Check-In Deduplication Window (Seconds)
            </label>
            <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-[10px] font-mono">Server-Controlled</span>
          </div>
          <input
            type="number"
            value={dedupWindow}
            onChange={(e) => setDedupWindow(parseInt(e.target.value))}
            className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-lg text-sm text-slate-900 focus:outline-none focus:border-[#007A99] font-mono"
          />
          <p className="text-xs text-slate-500">Time interval (AUTOROLL_ATTENDANCE_COOLDOWN_SECONDS=30) during which duplicate check-ins are suppressed.</p>
        </div>


        <button
          type="submit"
          className="w-full py-3 bg-[#007FFF] hover:bg-[#005FCC] text-white font-semibold rounded-lg shadow-sm flex items-center justify-center gap-2 transition-all text-sm"
        >
          <Save className="w-4 h-4" />
          <span>Save System Configuration</span>
        </button>
      </form>
    </div>
  );
};
