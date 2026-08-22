import {
  Camera,
  CameraCreatePayload,
  WorkerNode,
  Student,
  StudentCreatePayload,
  AttendanceRecord,
  DashboardMetrics,
} from '../types';

const getApiBase = (): string => {
  return '/api/v1';
};

export class ApiService {
  private static baseUrl = getApiBase();

  static async getDashboardMetrics(): Promise<DashboardMetrics> {
    const res = await fetch(`${this.baseUrl}/metrics/dashboard`);
    if (!res.ok) {
      throw new Error(`Failed to fetch dashboard metrics: ${res.statusText}`);
    }
    const data = await res.json();
    return {
      studentsTotal: data.students_total ?? 0,
      embeddingsEnrolled: data.embeddings_enrolled ?? 0,
      camerasTotal: data.cameras_total ?? 0,
      camerasActive: data.cameras_active ?? 0,
      workersTotal: data.workers_total ?? 0,
      workersOnline: data.workers_online ?? 0,
      attendanceToday: data.attendance_today ?? 0,
      recognitionFps: data.recognition_fps ?? null,
      avgLatencyMs: data.avg_latency_ms ?? null,
      p95LatencyMs: data.p95_latency_ms ?? null,
      spoofAttempts: data.spoof_attempts ?? null,
      unknownFaces: data.unknown_faces ?? null,
    };
  }

  static async getWorkers(): Promise<WorkerNode[]> {
    const res = await fetch(`${this.baseUrl}/workers`);
    if (!res.ok) {
      throw new Error(`Failed to fetch worker nodes: ${res.statusText}`);
    }
    const list = await res.json();
    if (!Array.isArray(list)) return [];
    return list.map((w: any) => ({
      id: w.worker_id || w.id || 'worker_unknown',
      host: w.host || undefined,
      state: w.state || 'UNKNOWN',
      cpuPercent: w.cpu_percent ?? w.cpuPercent ?? null,
      ramUsedMb: w.ram_used_mb ?? w.ramUsedMb ?? null,
      ramPercent: w.ram_percent ?? w.ramPercent ?? null,
      gpuAvailable: w.gpu_available ?? w.gpuAvailable ?? false,
      gpuName: w.gpu_name ?? w.gpuName ?? null,
      gpuUtilizationPercent: w.gpu_utilization_percent ?? w.gpuUtilizationPercent ?? null,
      gpuMemoryUsedMb: w.gpu_memory_used_mb ?? w.gpuMemoryUsedMb ?? null,
      activeCamerasCount: w.assigned_cameras_count ?? w.active_cameras_count ?? w.activeCamerasCount ?? 0,
      fps: w.fps ?? null,
      avgInferenceLatencyMs: w.avg_inference_latency_ms ?? w.avgInferenceLatencyMs ?? null,
      modelVersion: w.model_version || w.modelVersion || null,
      lastHeartbeat: w.last_heartbeat || null,
    }));
  }

  static async getCameras(): Promise<Camera[]> {
    const res = await fetch(`${this.baseUrl}/cameras`);
    if (!res.ok) {
      throw new Error(`Failed to fetch cameras: ${res.statusText}`);
    }
    const list = await res.json();
    if (!Array.isArray(list)) return [];
    return list.map((c: any) => ({
      id: c.camera_id || c.id,
      name: c.name,
      rtspUrl: c.rtsp_url || c.rtspUrl || '',
      location: c.location || null,
      targetFps: c.target_fps || 30,
      isActive: Boolean(c.is_active ?? true),
      assignedWorkerId: c.assigned_worker_id || null,
    }));
  }

  static async createCamera(payload: CameraCreatePayload): Promise<Camera> {
    const res = await fetch(`${this.baseUrl}/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: payload.name,
        rtsp_url: payload.rtspUrl,
        location: payload.location || null,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to create camera: ${res.statusText}`);
    }
    const data = await res.json();
    return {
      id: data.camera_id,
      name: data.name,
      rtspUrl: data.rtsp_url,
      location: payload.location || null,
      isActive: true,
      assignedWorkerId: null,
    };
  }

  static async assignCamera(cameraId: string, workerId?: string): Promise<boolean> {
    const res = await fetch(`${this.baseUrl}/cameras/${cameraId}/assign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ worker_id: workerId || null }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to assign camera: ${res.statusText}`);
    }
    return true;
  }

  static async unassignCamera(cameraId: string): Promise<boolean> {
    const res = await fetch(`${this.baseUrl}/cameras/${cameraId}/unassign`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to unassign camera: ${res.statusText}`);
    }
    return true;
  }

  static async getStudents(skip: number = 0, limit: number = 100): Promise<Student[]> {
    const res = await fetch(`${this.baseUrl}/students?skip=${skip}&limit=${limit}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch students: ${res.statusText}`);
    }
    const list = await res.json();
    if (!Array.isArray(list)) return [];
    return list.map((s: any) => ({
      id: s.id,
      studentCode: s.student_code,
      fullName: s.full_name,
      department: s.department || null,
      isActive: Boolean(s.is_active ?? true),
      hasEmbedding: true,
    }));
  }

  static async createStudent(payload: StudentCreatePayload): Promise<Student> {
    const res = await fetch(`${this.baseUrl}/students`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        student_code: payload.studentCode,
        full_name: payload.fullName,
        department: payload.department || null,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to create student: ${res.statusText}`);
    }
    const s = await res.json();
    return {
      id: s.id,
      studentCode: s.student_code,
      fullName: s.full_name,
      department: s.department || null,
      isActive: s.is_active,
    };
  }

  static async deleteStudent(studentId: string): Promise<boolean> {
    const res = await fetch(`${this.baseUrl}/students/${studentId}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to delete student: ${res.statusText}`);
    }
    return true;
  }


  static async enrollStudentFace(studentId: string, embedding: number[], modelVersion: string = 'arcface_iresnet50_v1'): Promise<boolean> {
    const res = await fetch(`${this.baseUrl}/students/${studentId}/enroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        embedding,
        model_version: modelVersion,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to enroll student face: ${res.statusText}`);
    }
    return true;
  }

  static async getAttendance(limit: number = 50): Promise<AttendanceRecord[]> {
    const res = await fetch(`${this.baseUrl}/attendance?limit=${limit}`);
    if (!res.ok) {
      throw new Error(`Failed to fetch attendance logs: ${res.statusText}`);
    }
    const list = await res.json();
    if (!Array.isArray(list)) return [];
    return list.map((r: any) => ({
      id: r.id,
      studentId: r.student_id,
      studentCode: r.student_code || undefined,
      fullName: r.full_name || undefined,
      cameraId: r.camera_id || undefined,
      cameraName: r.camera_name || undefined,
      workerId: r.worker_id || undefined,
      timestamp: r.timestamp,
      similarityScore: r.similarity_score ?? 0,
      livenessScore: r.liveness_score ?? 0,
      modelVersion: r.model_version || undefined,
      verificationStatus: r.verification_status || 'CONFIRMED',
    }));
  }

  static async checkHealth(): Promise<boolean> {
    try {
      const url = typeof window !== 'undefined' && window.location.port === '3000'
        ? 'http://localhost:8000/health'
        : '/health';
      const res = await fetch(url);
      return res.ok;
    } catch {
      return false;
    }
  }

  static async getSchedulerStatus(): Promise<any> {
    const res = await fetch(`${this.baseUrl}/scheduler/status`);
    if (!res.ok) {
      throw new Error(`Failed to fetch scheduler status: ${res.statusText}`);
    }
    return await res.json();
  }

  static async triggerRebalance(): Promise<any> {
    const res = await fetch(`${this.baseUrl}/scheduler/rebalance`, {
      method: 'POST',
    });
    if (!res.ok) {
      throw new Error(`Failed to trigger workload rebalance: ${res.statusText}`);
    }
    return await res.json();
  }
}
