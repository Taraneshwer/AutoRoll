export type UserRole = 'ADMIN' | 'MANAGER' | 'VIEWER';

export interface User {
  id: string;
  username: string;
  email?: string;
  role: UserRole;
  fullName?: string;
}

export interface Student {
  id: string;
  studentCode: string;
  fullName: string;
  department?: string | null;
  isActive: boolean;
  hasEmbedding?: boolean;
}

export interface StudentCreatePayload {
  studentCode: string;
  fullName: string;
  department?: string;
}

export interface Camera {
  id: string;
  name: string;
  rtspUrl: string;
  location?: string | null;
  targetFps?: number;
  isActive: boolean;
  assignedWorkerId?: string | null;
}

export interface CameraCreatePayload {
  name: string;
  rtspUrl: string;
  location?: string;
}

export interface WorkerNode {
  id: string;
  host?: string;
  state: 'STARTING' | 'READY' | 'BUSY' | 'DEGRADED' | 'OFFLINE' | 'STOPPING' | string;
  cpuPercent?: number | null;
  ramUsedMb?: number | null;
  ramPercent?: number | null;
  gpuAvailable?: boolean;
  gpuName?: string | null;
  gpuUtilizationPercent?: number | null;
  gpuMemoryUsedMb?: number | null;
  activeCamerasCount: number;
  fps?: number | null;
  avgInferenceLatencyMs?: number | null;
  modelVersion?: string | null;
  lastHeartbeat?: string | number | null;
  queueDepth?: number;
}


export interface AttendanceRecord {
  id: string;
  studentId: string;
  studentCode?: string;
  fullName?: string;
  cameraId?: string;
  cameraName?: string;
  workerId?: string;
  timestamp: string;
  similarityScore: number;
  livenessScore: number;
  modelVersion?: string;
  verificationStatus: 'CONFIRMED' | 'SPOOF_ATTEMPT' | 'UNKNOWN_PERSON' | string;
}

export interface DashboardMetrics {
  studentsTotal: number;
  embeddingsEnrolled: number;
  camerasTotal: number;
  camerasActive: number;
  workersTotal: number;
  workersOnline: number;
  attendanceToday: number;
  recognitionFps?: number | null;
  avgLatencyMs?: number | null;
  p95LatencyMs?: number | null;
  spoofAttempts?: number | null;
  unknownFaces?: number | null;
}

export type WebSocketStatus = 'CONNECTED' | 'CONNECTING' | 'DISCONNECTED' | 'RECONNECTING' | 'ERROR';

export interface WebSocketEventEnvelope {
  event_id: string;
  event_type: string;
  timestamp_ms: number;
  sequence_number: number;
  data: Record<string, any>;
}
