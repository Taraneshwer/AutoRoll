import React, { useState } from 'react';
import { ShieldCheck, Lock, User as UserIcon, ArrowRight } from 'lucide-react';
import { UserRole } from '../types';

interface LoginPageProps {
  onLoginSuccess: (user: { username: string; role: UserRole }) => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('password');
  const [role, setRole] = useState<UserRole>('ADMIN');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onLoginSuccess({ username, role });
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6 relative overflow-hidden">
      {/* Background Soft Gradients */}
      <div className="absolute top-1/4 left-1/3 w-96 h-96 bg-[#007FFF]/5 rounded-full blur-3xl pointer-events-none"></div>
      <div className="absolute bottom-1/4 right-1/3 w-96 h-96 bg-[#007FFF]/5 rounded-full blur-3xl pointer-events-none"></div>

      <div className="w-full max-w-md bg-white p-8 rounded-2xl border border-slate-200 shadow-xl relative z-10 animate-fade-in">
        {/* Brand Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-2xl bg-[#EAF5FF] border border-[#007FFF]/20 flex items-center justify-center mx-auto mb-4 shadow-xs">
            <ShieldCheck className="w-9 h-9 text-[#007FFF]" />
          </div>
          <h1 className="text-3xl font-extrabold text-[#007FFF] tracking-tight">
            AutoRoll
          </h1>
          <p className="text-sm text-slate-500 mt-1">Privacy-Preserving AI Attendance System</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">
              Username
            </label>
            <div className="relative">
              <UserIcon className="w-5 h-5 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="w-full pl-11 pr-4 py-2.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-[#007FFF] transition-colors text-sm"
                placeholder="Enter username"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">
              Password
            </label>
            <div className="relative">
              <Lock className="w-5 h-5 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full pl-11 pr-4 py-2.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-[#007FFF] transition-colors text-sm"
                placeholder="Enter password"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2">
              Role Access Scope
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['ADMIN', 'MANAGER', 'VIEWER'] as UserRole[]).map((r) => (
                <button
                  type="button"
                  key={r}
                  onClick={() => setRole(r)}
                  className={`py-2 text-xs font-bold rounded-lg border transition-all ${
                    role === r
                      ? 'bg-[#EAF5FF] text-[#007FFF] border-[#007FFF]/40 shadow-xs'
                      : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          <button
            type="submit"
            className="w-full py-3 bg-[#007FFF] hover:bg-[#005FCC] text-white font-bold rounded-lg shadow-sm flex items-center justify-center gap-2 transition-all group mt-2 text-sm"
          >
            <span>Sign In to Control Plane</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
        </form>

        <p className="text-center text-xs text-slate-500 mt-6">
          AutoRoll Privacy Guarantee: Raw face images are never stored or transmitted to the UI.
        </p>
      </div>
    </div>
  );
};
