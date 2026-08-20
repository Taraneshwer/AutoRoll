import React, { useState } from 'react';
import { UserCheck, Search, Building, AlertTriangle, UserX, RefreshCw, Trash2, ShieldAlert } from 'lucide-react';
import { useStudents } from '../hooks/useAutoRollData';

export const StudentsPage: React.FC = () => {
  const { students, loading, error, refresh } = useStudents();
  const [search, setSearch] = useState('');
  const [deletingStudent, setDeletingStudent] = useState<{ id: string; name: string } | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);

  const filtered = students.filter(
    (s) =>
      s.fullName?.toLowerCase().includes(search.toLowerCase()) ||
      s.studentCode?.toLowerCase().includes(search.toLowerCase()) ||
      s.department?.toLowerCase().includes(search.toLowerCase())
  );

  const confirmDelete = async () => {
    if (!deletingStudent) return;
    setIsDeleting(true);
    setDeleteError(null);

    try {
      const res = await fetch(`/api/v1/students/${deletingStudent.id}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        throw new Error('Failed to delete student.');
      }
      setDeletingStudent(null);
      refresh();
    } catch (err: any) {
      setDeleteError(err.message || 'Error deleting student.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Student Directory</h2>
          <p className="text-sm text-slate-500 mt-1">Enrolled student identity database with ArcFace embedding status</p>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <button
            onClick={refresh}
            className="px-3 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-[#EAF5FF] hover:text-[#007FFF] text-xs font-semibold flex items-center gap-1.5 transition-all shadow-xs"
            title="Refresh Student Directory"
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
              placeholder="Search students..."
              className="w-full pl-10 pr-4 py-2 bg-white border border-slate-200 rounded-xl text-xs text-slate-900 focus:outline-none focus:border-[#007FFF]"
            />
          </div>
        </div>
      </div>

      {loading && students.length === 0 ? (
        <div className="bg-white p-12 rounded-2xl border border-slate-200 text-center text-slate-500 text-sm">
          Loading student directory...
        </div>
      ) : error ? (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700 flex items-center justify-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-600" />
          <span>Unable to load student directory: {error}</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-white p-12 rounded-2xl text-center text-slate-500 text-sm flex flex-col items-center justify-center gap-3 border border-dashed border-slate-200">
          <UserX className="w-10 h-10 text-slate-400" />
          <p className="font-bold text-slate-800">No Students Enrolled</p>
          <p className="text-xs text-slate-500">
            {search ? 'No students match your search filter.' : 'Use the Face Enrollment tab to enroll new students.'}
          </p>
        </div>
      ) : (
        /* Grid of Student Cards */
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {filtered.map((s) => (
            <div key={s.id} className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs space-y-4 relative hover:border-[#007FFF] transition-all">
              <div className="flex items-center justify-between">
                <div className="w-12 h-12 rounded-xl bg-[#EAF5FF] border border-[#007FFF]/20 flex items-center justify-center font-extrabold text-[#007FFF] text-lg">
                  {s.fullName?.charAt(0).toUpperCase() || 'S'}
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2.5 py-1 text-[10px] font-bold rounded-full ${
                    s.isActive
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : 'bg-slate-100 text-slate-600 border border-slate-200'
                  } flex items-center gap-1`}>
                    <UserCheck className="w-3 h-3" />
                    {s.isActive ? 'ACTIVE' : 'INACTIVE'}
                  </span>
                  <button
                    onClick={() => setDeletingStudent({ id: s.id, name: s.fullName })}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
                    title="Delete Student"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div>
                <h3 className="font-bold text-slate-900 text-base">{s.fullName}</h3>
                <p className="text-xs font-mono text-[#007FFF] font-medium mt-0.5">{s.studentCode}</p>
              </div>

              <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                <div className="flex items-center gap-1.5">
                  <Building className="w-3.5 h-3.5 text-slate-400" />
                  <span>{s.department || 'General'}</span>
                </div>
                <span className="text-[10px] text-slate-400 font-mono">512-d Template</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Confirmation Modal */}
      {deletingStudent && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xl max-w-md w-full space-y-4 text-center">
            <div className="w-12 h-12 bg-rose-50 border border-rose-200 rounded-2xl flex items-center justify-center mx-auto text-rose-600">
              <ShieldAlert className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-900">Confirm Student Deletion</h3>
            <p className="text-xs text-slate-500">
              Are you sure you want to delete <strong className="text-slate-900">{deletingStudent.name}</strong>? <strong>This will permanently delete the student's biometric template.</strong>
            </p>


            {deleteError && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700">
                {deleteError}
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <button
                onClick={() => setDeletingStudent(null)}
                className="flex-1 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold rounded-xl text-xs transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                disabled={isDeleting}
                className="flex-1 py-2.5 bg-rose-600 hover:bg-rose-700 text-white font-bold rounded-xl text-xs transition-colors disabled:opacity-50"
              >
                {isDeleting ? 'Deleting...' : 'Delete Student'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
