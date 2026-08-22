import React, { useState } from 'react';
import { UserPlus, Camera, CheckCircle2, AlertCircle, Lock, RefreshCw, ArrowRight, ShieldCheck } from 'lucide-react';

interface EnrollmentSummary {
  student_id: string;
  student_code: string;
  full_name: string;
  department?: string;
  model_id: string;
  model_version: string;
  embedding_dimension: number;
  sample_count: number;
  status: string;
}

export const EnrollmentPage: React.FC = () => {
  const [step, setStep] = useState<number>(1);
  const [studentCode, setStudentCode] = useState('');
  const [fullName, setFullName] = useState('');
  const [department, setDepartment] = useState('');

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sampleCount, setSampleCount] = useState<number>(0);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [reasons, setReasons] = useState<string[]>([]);
  const [summary, setSummary] = useState<EnrollmentSummary | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  // Step 1: Start Enrollment Session
  const handleStartSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentCode.trim() || !fullName.trim()) return;

    setErrorMsg('');
    try {
      const res = await fetch('/api/v1/enrollment/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_code: studentCode.trim(),
          full_name: fullName.trim(),
          department: department.trim() || undefined,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to start enrollment session.');
      }

      const data = await res.json();
      setSessionId(data.session_id);
      setStep(2);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start session.');
    }
  };

  // Step 4: Capture Frame Sample from Live Camera Stream
  const handleCaptureFrame = async () => {
    if (!sessionId) return;

    setIsCapturing(true);
    setErrorMsg('');

    try {
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');

      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = '/api/v1/camera/mjpeg?' + new Date().getTime();

      await new Promise((resolve) => {
        img.onload = () => {
          if (ctx) ctx.drawImage(img, 0, 0, 640, 480);
          resolve(true);
        };
        img.onerror = () => resolve(false);
      });

      canvas.toBlob(async (blob) => {
        if (!blob) {
          setErrorMsg('Failed to grab camera frame.');
          setIsCapturing(false);
          return;
        }

        const formData = new FormData();
        formData.append('file', blob, 'enrollment_frame.jpg');

        const res = await fetch(`/api/v1/enrollment/${sessionId}/frame`, {
          method: 'POST',
          body: formData,
        });

        const data = await res.json();
        if (data.accepted) {
          setSampleCount(data.sample_count);
          if (data.sample_count >= 5) {
            setStep(5);
          }
        } else {
          setReasons((prev) => [data.reason || 'Rejected frame', ...prev.slice(0, 4)]);
        }
        setIsCapturing(false);
      }, 'image/jpeg');
    } catch (err: any) {
      setErrorMsg('Failed to process enrollment frame.');
      setIsCapturing(false);
    }
  };

  // Step 5: Complete Enrollment
  const handleCompleteEnrollment = async () => {
    if (!sessionId) return;
    setErrorMsg('');

    try {
      const res = await fetch(`/api/v1/enrollment/${sessionId}/complete`, {
        method: 'POST',
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to complete enrollment.');
      }

      const data: EnrollmentSummary = await res.json();
      setSummary(data);
      setStep(6);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to complete template aggregation.');
    }
  };

  const handleReset = () => {
    setStep(1);
    setStudentCode('');
    setFullName('');
    setDepartment('');
    setSessionId(null);
    setSampleCount(0);
    setReasons([]);
    setSummary(null);
    setErrorMsg('');
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <UserPlus className="w-6 h-6 text-[#007FFF]" />
          Student Face Enrollment Wizard
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          Multi-sample capture (5–10 samples), quality & liveness verification, and normalized mean template generation
        </p>
      </div>

      {/* Progress Wizard Steps Indicator */}
      <div className="grid grid-cols-6 gap-2 text-center text-xs font-bold">
        {['Info', 'Camera', 'Target', 'Capture', 'Template', 'Complete'].map((stName, idx) => {
          const sNum = idx + 1;
          const isActive = step === sNum;
          const isDone = step > sNum;
          return (
            <div
              key={sNum}
              className={`p-2.5 rounded-xl border transition-all ${
                isActive
                  ? 'bg-[#EAF5FF] text-[#007FFF] border-[#007FFF]'
                  : isDone
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : 'bg-white text-slate-400 border-slate-200'
              }`}
            >
              Step {sNum}: {stName}
            </div>
          );
        })}
      </div>

      {/* Zero Permanent Image Storage Notice */}
      <div className="p-4 bg-[#EAF5FF] border border-[#007FFF]/30 rounded-xl flex items-center gap-3 text-xs text-[#005FCC]">
        <Lock className="w-5 h-5 text-[#007FFF] shrink-0" />
        <div>
          <span className="font-bold">Zero Raw Photographs Persisted:</span> Face chips are processed transiently in memory to generate 512-dim ArcFace normalized templates. Raw images are purged immediately.
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-center gap-2 text-xs text-rose-700 font-semibold">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* STEP 1: Student Info Form */}
      {step === 1 && (
        <form onSubmit={handleStartSession} className="space-y-6 bg-white p-6 rounded-2xl border border-slate-200 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-3">Step 1: Student Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div>
              <label className="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2">Student Code *</label>
              <input
                type="text"
                value={studentCode}
                onChange={(e) => setStudentCode(e.target.value)}
                placeholder="e.g. STU2001"
                required
                className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-mono text-slate-900 focus:outline-none focus:border-[#007FFF]"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2">Full Name *</label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="e.g. Sarah Jenkins"
                required
                className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm text-slate-900 focus:outline-none focus:border-[#007FFF]"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2">Department</label>
              <input
                type="text"
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                placeholder="e.g. Computer Science"
                className="w-full px-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm text-slate-900 focus:outline-none focus:border-[#007FFF]"
              />
            </div>
          </div>

          <button
            type="submit"
            className="w-full py-3 bg-[#007FFF] hover:bg-[#005FCC] text-white font-bold rounded-xl shadow-xs flex items-center justify-center gap-2 transition-all text-sm"
          >
            <span>Start Enrollment Session</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>
      )}

      {/* STEP 2 & 3: Camera Readiness & Position Target */}
      {(step === 2 || step === 3) && (
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs space-y-6 text-center">
          <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-3 text-left">
            Step {step}: Camera Readiness & Face Target Box
          </h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Position student's face inside the camera view, ensuring adequate lighting and zero motion blur.
          </p>

          <div className="relative aspect-video max-w-lg mx-auto bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center">
            <img src="/api/v1/camera/mjpeg" alt="Webcam Stream" className="w-full h-full object-contain" />
            <div className="absolute inset-0 border-4 border-dashed border-[#007FFF]/60 rounded-2xl pointer-events-none"></div>
          </div>

          <button
            onClick={() => setStep(4)}
            className="px-8 py-3 bg-[#007FFF] hover:bg-[#005FCC] text-white font-bold rounded-xl shadow-xs inline-flex items-center gap-2 transition-all text-sm"
          >
            <span>Proceed to Face Capture</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* STEP 4: Sample Capture Workflow */}
      {step === 4 && (
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs space-y-6">
          <div className="flex justify-between items-center border-b border-slate-100 pb-3">
            <h3 className="text-sm font-bold text-slate-900">Step 4: Sample Capture ({sampleCount} / 5 required)</h3>
            <span className="text-xs font-mono font-bold text-[#007FFF]">Session: {sessionId}</span>
          </div>

          {/* Sample Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-bold text-slate-600">
              <span>Enrollment Samples Collected</span>
              <span>{Math.min(100, (sampleCount / 5) * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#007FFF] transition-all duration-300 rounded-full"
                style={{ width: `${Math.min(100, (sampleCount / 5) * 100)}%` }}
              ></div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center">
              <img src="/api/v1/camera/mjpeg" alt="Webcam Stream" className="w-full h-full object-contain" />
            </div>

            <div className="space-y-4 flex flex-col justify-between">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Recent Filter Rejections</h4>
                {reasons.length === 0 ? (
                  <p className="text-xs text-slate-400 font-mono">No rejection events logged yet.</p>
                ) : (
                  <div className="space-y-2">
                    {reasons.map((r, idx) => (
                      <span key={idx} className="block px-3 py-1.5 bg-rose-50 border border-rose-200 rounded-xl text-xs font-mono text-rose-700">
                        Rejected: {r}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <button
                onClick={handleCaptureFrame}
                disabled={isCapturing}
                className="w-full py-3 bg-[#007FFF] hover:bg-[#005FCC] text-white font-bold rounded-xl shadow-xs flex items-center justify-center gap-2 transition-all text-sm disabled:opacity-50"
              >
                <Camera className="w-4 h-4" />
                <span>{isCapturing ? 'Analyzing & Extracting...' : 'Capture Valid Sample'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* STEP 5: Mean Template Aggregation */}
      {step === 5 && (
        <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-xs text-center space-y-6">
          <div className="w-16 h-16 bg-[#EAF5FF] border border-[#007FFF]/30 rounded-2xl flex items-center justify-center mx-auto text-[#007FFF]">
            <ShieldCheck className="w-8 h-8" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-slate-900">Step 5: Compute Normalized Mean Template</h3>
            <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
              Collected {sampleCount} valid face samples. Aggregating 512-dimensional normalized mean embedding template for DB storage.
            </p>
          </div>

          <button
            onClick={handleCompleteEnrollment}
            className="px-8 py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-xl shadow-xs inline-flex items-center gap-2 transition-all text-sm"
          >
            <span>Aggregate & Complete Enrollment</span>
            <CheckCircle2 className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* STEP 6: Success Summary */}
      {step === 6 && summary && (
        <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-xs space-y-6 text-center">
          <div className="w-16 h-16 bg-emerald-50 border border-emerald-200 rounded-2xl flex items-center justify-center mx-auto text-emerald-600">
            <CheckCircle2 className="w-8 h-8" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-slate-900">Student Enrolled Successfully</h3>
            <p className="text-xs text-slate-500 mt-1">Face template stored in DB with strict model versioning metadata</p>
          </div>

          <div className="max-w-md mx-auto p-4 bg-slate-50 rounded-2xl border border-slate-200 space-y-2.5 text-left text-xs font-mono">
            <div className="flex justify-between border-b border-slate-200 pb-1.5">
              <span className="text-slate-500">Student Code:</span>
              <strong className="text-slate-900">{summary.student_code}</strong>
            </div>
            <div className="flex justify-between border-b border-slate-200 pb-1.5">
              <span className="text-slate-500">Full Name:</span>
              <strong className="text-slate-900">{summary.full_name}</strong>
            </div>
            <div className="flex justify-between border-b border-slate-200 pb-1.5">
              <span className="text-slate-500">Samples Used:</span>
              <strong className="text-emerald-700">{summary.sample_count} samples</strong>
            </div>
            <div className="flex justify-between border-b border-slate-200 pb-1.5">
              <span className="text-slate-500">Model ID:</span>
              <strong className="text-[#007FFF]">{summary.model_id}</strong>
            </div>
            <div className="flex justify-between border-b border-slate-200 pb-1.5">
              <span className="text-slate-500">Model Version:</span>
              <strong className="text-slate-800">{summary.model_version}</strong>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Vector Dimension:</span>
              <strong className="text-purple-700">{summary.embedding_dimension}-d normalized</strong>
            </div>
          </div>

          <button
            onClick={handleReset}
            className="px-8 py-3 bg-[#007FFF] hover:bg-[#005FCC] text-white font-bold rounded-xl shadow-xs inline-flex items-center gap-2 transition-all text-sm"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Enroll Another Student</span>
          </button>
        </div>
      )}
    </div>
  );
};
