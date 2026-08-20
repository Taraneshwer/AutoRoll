import React, { useState } from 'react';
import { Camera as CameraIcon, Cpu, Plus, Link, AlertTriangle, VideoOff, X } from 'lucide-react';
import { ApiService } from '../services/api';
import { useCameras, useWorkers } from '../hooks/useAutoRollData';

export const CamerasPage: React.FC = () => {
  const { cameras, setCameras, loading, error, refresh } = useCameras();
  const { workers } = useWorkers();

  const [showModal, setShowModal] = useState(false);
  const [name, setName] = useState('');
  const [rtspUrl, setRtspUrl] = useState('');
  const [location, setLocation] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [actionError, setActionError] = useState('');

  const handleAssignmentChange = async (cameraId: string, workerId: string) => {
    setActionError('');
    try {
      if (workerId === 'NONE') {
        await ApiService.unassignCamera(cameraId);
        setCameras((prev) =>
          prev.map((cam) => (cam.id === cameraId ? { ...cam, assignedWorkerId: null } : cam))
        );
      } else {
        await ApiService.assignCamera(cameraId, workerId);
        setCameras((prev) =>
          prev.map((cam) => (cam.id === cameraId ? { ...cam, assignedWorkerId: workerId } : cam))
        );
      }
    } catch (err: any) {
      setActionError(err.message || 'Failed to update camera assignment');
    }
  };

  const handleAddCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !rtspUrl.trim()) return;

    setIsSubmitting(true);
    setActionError('');
    try {
      await ApiService.createCamera({
        name: name.trim(),
        rtspUrl: rtspUrl.trim(),
        location: location.trim() || undefined,
      });
      setName('');
      setRtspUrl('');
      setLocation('');
      setShowModal(false);
      refresh();
    } catch (err: any) {
      setActionError(err.message || 'Failed to create camera');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">RTSP Camera Management</h2>
          <p className="text-sm text-slate-500 mt-1">Configure stream endpoints and assign worker cluster processing nodes</p>
        </div>

        <button
          onClick={() => setShowModal(true)}
          className="px-4 py-2 bg-[#007FFF] hover:bg-[#005FCC] text-white text-xs font-semibold rounded-lg shadow-xs flex items-center gap-2 transition-all"
        >
          <Plus className="w-4 h-4" />
          <span>Add New Camera</span>
        </button>
      </div>

      {actionError && (
        <div className="p-4 bg-[#FEF2F2] border border-[#FECACA] rounded-xl text-xs text-[#B91C1C] flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-[#DC2626] shrink-0" />
          <span>{actionError}</span>
        </div>
      )}

      {loading && cameras.length === 0 ? (
        <div className="bg-white p-12 rounded-xl border border-slate-200 text-center text-slate-500 text-sm">
          Loading camera configuration...
        </div>
      ) : error ? (
        <div className="p-4 bg-[#FEF2F2] border border-[#FECACA] rounded-xl text-xs text-[#B91C1C] flex items-center justify-center gap-2">
          <AlertTriangle className="w-4 h-4 text-[#DC2626]" />
          <span>Unable to load cameras: {error}</span>
        </div>
      ) : cameras.length === 0 ? (
        <div className="bg-white p-12 rounded-xl text-center text-slate-500 text-sm flex flex-col items-center justify-center gap-3 border border-dashed border-slate-200">
          <VideoOff className="w-10 h-10 text-slate-400" />
          <p className="font-bold text-slate-800">No RTSP Cameras Configured</p>
          <p className="text-xs text-slate-500">
            Click "Add New Camera" above to register an RTSP stream endpoint.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {cameras.map((cam) => (
            <div key={cam.id} className="bg-white p-6 rounded-xl border border-slate-200 shadow-xs space-y-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-[#EAF5FF] border border-[#007FFF]/20 flex items-center justify-center text-[#007FFF]">
                    <CameraIcon className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-slate-900 text-base">{cam.name}</h3>
                    <p className="text-xs text-slate-500">{cam.location || 'Unspecified Location'}</p>
                  </div>
                </div>
                <span className={`px-2.5 py-1 text-[10px] font-bold rounded-full ${
                  cam.isActive ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-slate-100 text-slate-600'
                }`}>
                  {cam.isActive ? 'ACTIVE' : 'DISABLED'}
                </span>
              </div>

              {/* RTSP Stream URL */}
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 flex items-center justify-between text-xs font-mono text-slate-700">
                <div className="flex items-center gap-2 truncate">
                  <Link className="w-4 h-4 text-[#007FFF] shrink-0" />
                  <span className="truncate">{cam.rtspUrl}</span>
                </div>
                <span className="text-[10px] text-slate-500 font-sans ml-2 shrink-0">{cam.targetFps || 30} FPS Target</span>
              </div>

              {/* Worker Assignment Selector */}
              <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-600 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-[#007FFF]" />
                  <span>Worker Processing Node:</span>
                </label>

                <select
                  value={cam.assignedWorkerId || 'NONE'}
                  onChange={(e) => handleAssignmentChange(cam.id, e.target.value)}
                  className="bg-white text-slate-900 text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-200 focus:outline-none focus:border-[#007FFF] cursor-pointer"
                >
                  <option value="NONE">Unassigned (Offline)</option>
                  {workers.map((w) => (
                    <option key={w.id} value={w.id}>
                      {w.id} ({w.gpuAvailable ? w.gpuName || 'GPU' : 'CPU'})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add Camera Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-xl max-w-md w-full space-y-5 animate-fade-in relative">
            <div className="flex justify-between items-center border-b border-slate-100 pb-3">
              <h3 className="font-bold text-slate-900 text-lg">Add New RTSP Camera</h3>
              <button onClick={() => setShowModal(false)} className="text-slate-400 hover:text-slate-700">
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleAddCamera} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">
                  Camera Name *
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Main Entrance Lobby"
                  required
                  className="w-full px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-900 focus:outline-none focus:border-[#007FFF]"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">
                  RTSP Stream URL *
                </label>
                <input
                  type="text"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="rtsp://192.168.1.100:554/stream1"
                  required
                  className="w-full px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-900 focus:outline-none focus:border-[#007FFF] font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">
                  Location
                </label>
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="e.g. Building A - Gate 1"
                  className="w-full px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-900 focus:outline-none focus:border-[#007FFF]"
                />
              </div>

              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="flex-1 py-2 bg-white border border-slate-200 text-slate-700 font-semibold rounded-lg text-xs hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex-1 py-2 bg-[#007FFF] hover:bg-[#005FCC] text-white font-semibold rounded-lg text-xs shadow-xs disabled:opacity-50"
                >
                  {isSubmitting ? 'Registering...' : 'Register Camera'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
