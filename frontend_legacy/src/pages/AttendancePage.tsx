import React, { useState } from 'react';
import { Search, ShieldCheck, AlertTriangle, RefreshCw, ClipboardX } from 'lucide-react';
import { useAttendance } from '../hooks/useAutoRollData';

export const AttendancePage: React.FC = () => {
  const { records, loading, error, refresh } = useAttendance(100);
  const [search, setSearch] = useState('');

  const filtered = records.filter(
    (r) =>
      r.fullName?.toLowerCase().includes(search.toLowerCase()) ||
      r.studentCode?.toLowerCase().includes(search.toLowerCase()) ||
      r.studentId?.toLowerCase().includes(search.toLowerCase()) ||
      r.cameraName?.toLowerCase().includes(search.toLowerCase()) ||
      r.cameraId?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Attendance Log Records</h2>
          <p className="text-sm text-slate-500 mt-1">Verified live face attendance check-in audit stream</p>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <button
            onClick={refresh}
            className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 hover:bg-[#EAF5FF] hover:text-[#007FFF] hover:border-[#007FFF]/30 text-xs font-semibold flex items-center gap-1.5 transition-all shadow-xs"
            title="Refresh Attendance Log"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
          <div className="relative w-full md:w-72">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search student or camera..."
              className="w-full pl-10 pr-4 py-2 bg-white border border-slate-200 rounded-lg text-xs text-slate-900 focus:outline-none focus:border-[#007FFF] focus:ring-1 focus:ring-[#007FFF]"
            />
          </div>
        </div>
      </div>

      {loading && records.length === 0 ? (
        <div className="bg-white p-12 rounded-xl border border-slate-200 text-center text-slate-500 text-sm">
          Loading attendance records...
        </div>
      ) : error ? (
        <div className="p-4 bg-[#FEF2F2] border border-[#FECACA] rounded-xl text-xs text-[#B91C1C] flex items-center justify-center gap-2">
          <AlertTriangle className="w-4 h-4 text-[#DC2626]" />
          <span>Unable to load attendance telemetry: {error}</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-white p-12 rounded-xl text-center text-slate-500 text-sm flex flex-col items-center justify-center gap-3 border border-dashed border-slate-200">
          <ClipboardX className="w-10 h-10 text-slate-400" />
          <p className="font-bold text-slate-800">No Attendance Records Found</p>
          <p className="text-xs text-slate-500">
            {search ? 'No attendance records match your search filter.' : 'No face recognition check-ins recorded today.'}
          </p>
        </div>
      ) : (
        /* Attendance Table */
        <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
          <div className="overflow-x-auto">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Student ID</th>
                  <th>Student Code</th>
                  <th>Camera ID</th>
                  <th>Similarity</th>
                  <th>Liveness</th>
                  <th>Model Version</th>
                  <th>Status</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-[#EAF5FF] border border-[#007FFF]/20 flex items-center justify-center font-bold text-[#007FFF] text-xs">
                          {(r.fullName || r.studentCode || r.studentId).charAt(0).toUpperCase()}
                        </div>
                        <span className="font-semibold text-slate-900">{r.fullName || r.studentId}</span>
                      </div>
                    </td>
                    <td className="font-mono text-xs text-slate-600">{r.studentCode || r.studentId}</td>
                    <td className="text-slate-700">{r.cameraName || r.cameraId || 'N/A'}</td>
                    <td className="font-mono text-xs text-slate-900 font-medium">
                      {r.similarityScore !== undefined ? `${(r.similarityScore * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="font-mono text-xs text-emerald-600 font-medium">
                      {r.livenessScore !== undefined ? `${(r.livenessScore * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="font-mono text-xs text-slate-500">{r.modelVersion || 'arcface'}</td>
                    <td>
                      <span className="px-2.5 py-1 text-[10px] font-extrabold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 inline-flex items-center gap-1">
                        <ShieldCheck className="w-3 h-3" />
                        {r.verificationStatus}
                      </span>
                    </td>
                    <td className="text-xs text-slate-500 font-mono">
                      {r.timestamp ? new Date(r.timestamp).toLocaleString() : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
