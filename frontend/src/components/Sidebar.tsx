import React from 'react';
import {
  LayoutDashboard,
  Video,
  ClipboardList,
  Users,
  UserPlus,
  Camera as CameraIcon,
  Cpu,
  BrainCircuit,
  BarChart3,
  Settings as SettingsIcon,
  LogOut,
  ShieldCheck,
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  currentUser: { username: string; role: string } | null;
  onLogout: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  currentUser,
  onLogout,
}) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'monitoring', label: 'Live Monitoring', icon: Video },
    { id: 'attendance', label: 'Attendance Logs', icon: ClipboardList },
    { id: 'students', label: 'Student Directory', icon: Users },
    { id: 'enrollment', label: 'Face Enrollment', icon: UserPlus },
    { id: 'cameras', label: 'RTSP Cameras', icon: CameraIcon },
    { id: 'workers', label: 'ML Worker Nodes', icon: Cpu },
    { id: 'models', label: 'Model / ML Status', icon: BrainCircuit },
    { id: 'analytics', label: 'System Analytics', icon: BarChart3 },
    { id: 'settings', label: 'System Settings', icon: SettingsIcon },
  ];

  return (
    <aside className="w-[250px] bg-white border-r border-slate-200/90 shadow-[1px_0_6px_rgba(15,23,42,0.02)] flex flex-col h-screen sticky top-0 z-30 select-none shrink-0">
      {/* Brand Header */}
      <div className="p-4 pt-5 pb-4 border-b border-slate-100 flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#007FFF] to-[#005FCC] text-white flex items-center justify-center shrink-0 shadow-xs">
          <ShieldCheck className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-extrabold text-[#007FFF] leading-tight tracking-tight">
            AutoRoll
          </h1>
          <p className="text-[11px] text-slate-500 font-medium leading-none mt-0.5">Distributed AI Edge</p>
        </div>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 px-3 py-3 overflow-y-auto space-y-1">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full h-[38px] px-3 rounded-lg text-xs md:text-sm font-medium flex items-center gap-2.5 transition-all duration-150 ${
                isActive
                  ? 'bg-gradient-to-r from-[#EAF5FF] to-[#F1F7FF] text-[#007FFF] border-l-[3px] border-[#007FFF] font-bold shadow-2xs'
                  : 'text-slate-600 hover:text-[#007FFF] hover:bg-[#EAF5FF]/50'
              }`}
            >
              <Icon className={`w-4 h-4 shrink-0 transition-colors ${isActive ? 'text-[#007FFF]' : 'text-slate-400'}`} />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* User Profile Footer */}
      <div className="h-[72px] px-4 border-t border-slate-200/80 bg-slate-50/70 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#007FFF] to-[#005FCC] flex items-center justify-center font-bold text-white text-xs shrink-0 shadow-2xs">
            {currentUser?.username?.charAt(0).toUpperCase() || 'A'}
          </div>
          <div className="truncate">
            <p className="text-xs font-bold text-slate-900 truncate leading-tight">
              {currentUser?.username || 'Administrator'}
            </p>
            <span className="inline-flex items-center gap-1 text-[9px] font-extrabold tracking-wider uppercase rounded px-1.5 py-0.2 bg-[#EAF5FF] text-[#007FFF] border border-[#007FFF]/20 mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              {currentUser?.role || 'ADMIN'}
            </span>
          </div>
        </div>
        <button
          onClick={onLogout}
          title="Logout"
          className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors shrink-0"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </aside>
  );
};
