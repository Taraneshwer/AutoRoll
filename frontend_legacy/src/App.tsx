import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { LiveMonitoringPage } from './pages/LiveMonitoringPage';
import { AttendancePage } from './pages/AttendancePage';
import { StudentsPage } from './pages/StudentsPage';
import { EnrollmentPage } from './pages/EnrollmentPage';
import { CamerasPage } from './pages/CamerasPage';
import { WorkersPage } from './pages/WorkersPage';
import { ModelsPage } from './pages/ModelsPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { SettingsPage } from './pages/SettingsPage';
import { UserRole } from './types';

export const App: React.FC = () => {
  const [user, setUser] = useState<{ username: string; role: UserRole } | null>({
    username: 'admin',
    role: 'ADMIN',
  });
  const [activeTab, setActiveTab] = useState<string>('dashboard');

  if (!user) {
    return <LoginPage onLoginSuccess={(u) => setUser(u)} />;
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardPage />;
      case 'monitoring':
        return <LiveMonitoringPage />;
      case 'attendance':
        return <AttendancePage />;
      case 'students':
        return <StudentsPage />;
      case 'enrollment':
        return <EnrollmentPage />;
      case 'cameras':
        return <CamerasPage />;
      case 'workers':
        return <WorkersPage />;
      case 'models':
        return <ModelsPage />;
      case 'analytics':
        return <AnalyticsPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <DashboardPage />;
    }
  };

  const pageTitles: Record<string, string> = {
    dashboard: 'Dashboard',
    monitoring: 'Live Monitoring',
    attendance: 'Attendance Logs',
    students: 'Student Directory',
    enrollment: 'Face Enrollment',
    cameras: 'RTSP Cameras',
    workers: 'ML Worker Nodes',
    models: 'Model / ML Status',
    analytics: 'Analytics',
    settings: 'Settings',
  };

  return (
    <div className="flex min-h-screen bg-[#F8FAFC] text-slate-900">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentUser={user}
        onLogout={() => setUser(null)}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <Navbar activeTabTitle={pageTitles[activeTab] || 'Dashboard'} />
        <main className="flex-1 px-7 pt-6 pb-8 overflow-y-auto">{renderContent()}</main>
      </div>
    </div>
  );
};

export default App;
