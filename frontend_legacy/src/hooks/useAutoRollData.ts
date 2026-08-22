import { useState, useEffect, useCallback } from 'react';
import { ApiService } from '../services/api';
import { WebSocketService } from '../services/websocket';
import {
  DashboardMetrics,
  WorkerNode,
  Camera,
  AttendanceRecord,
  Student,
  WebSocketStatus,
  WebSocketEventEnvelope,
} from '../types';

export function useWebSocketStatus() {
  const [status, setStatus] = useState<WebSocketStatus>('DISCONNECTED');

  useEffect(() => {
    const ws = WebSocketService.getInstance();
    ws.connect();
    const unsubscribe = ws.onStatusChange(setStatus);
    return () => {
      unsubscribe();
    };
  }, []);

  return {
    status,
    isConnected: status === 'CONNECTED',
  };
}

export function useDashboardMetrics() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.getDashboardMetrics();
      setMetrics(data);
    } catch (err: any) {
      setError(err.message || 'Unable to connect to backend server');
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();

    // Subscribe to WebSocket events for real-time telemetry updates
    const ws = WebSocketService.getInstance();
    ws.connect();

    const unsubMetrics = ws.subscribe('SYSTEM_METRICS_UPDATED', fetchMetrics);
    const unsubAtt = ws.subscribe('ATTENDANCE_CONFIRMED', fetchMetrics);
    const unsubWorker = ws.subscribe('WORKER_STATUS_CHANGED', fetchMetrics);
    const unsubCamera = ws.subscribe('CAMERA_STATUS_CHANGED', fetchMetrics);

    return () => {
      unsubMetrics();
      unsubAtt();
      unsubWorker();
      unsubCamera();
    };
  }, [fetchMetrics]);

  return { metrics, loading, error, refresh: fetchMetrics };
}

export function useWorkers() {
  const [workers, setWorkers] = useState<WorkerNode[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkers = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const list = await ApiService.getWorkers();
      setWorkers(list);
    } catch (err: any) {
      setError(err.message || 'Unable to load worker cluster nodes');
      setWorkers([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkers();

    const ws = WebSocketService.getInstance();
    ws.connect();

    const unsubWorker = ws.subscribe('WORKER_STATUS_CHANGED', (evt: WebSocketEventEnvelope) => {
      // Dynamic inline update or refetch
      if (evt.data && evt.data.worker_id) {
        setWorkers((prev) =>
          prev.map((w) =>
            w.id === evt.data.worker_id
              ? {
                  ...w,
                  state: evt.data.state || w.state,
                  cpuPercent: evt.data.cpu_percent ?? w.cpuPercent,
                  gpuUtilizationPercent: evt.data.gpu_utilization_percent ?? w.gpuUtilizationPercent,
                  fps: evt.data.fps ?? w.fps,
                  avgInferenceLatencyMs: evt.data.avg_inference_latency_ms ?? w.avgInferenceLatencyMs,
                }
              : w
          )
        );
      } else {
        fetchWorkers();
      }
    });

    return () => {
      unsubWorker();
    };
  }, [fetchWorkers]);

  return { workers, loading, error, refresh: fetchWorkers };
}

export function useCameras() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCameras = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const list = await ApiService.getCameras();
      setCameras(list);
    } catch (err: any) {
      setError(err.message || 'Unable to load camera streams');
      setCameras([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCameras();

    const ws = WebSocketService.getInstance();
    ws.connect();

    const unsubCamera = ws.subscribe('CAMERA_STATUS_CHANGED', fetchCameras);
    return () => {
      unsubCamera();
    };
  }, [fetchCameras]);

  return { cameras, setCameras, loading, error, refresh: fetchCameras };
}

export function useAttendance(limit: number = 50) {
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAttendance = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const list = await ApiService.getAttendance(limit);
      setRecords(list);
    } catch (err: any) {
      setError(err.message || 'Unable to load attendance log');
      setRecords([]);
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchAttendance();

    const ws = WebSocketService.getInstance();
    ws.connect();

    const unsubAtt = ws.subscribe('ATTENDANCE_CONFIRMED', (evt: WebSocketEventEnvelope) => {
      if (evt.data && evt.data.attendance_id) {
        const newRecord: AttendanceRecord = {
          id: evt.data.attendance_id,
          studentId: evt.data.student_id,
          studentCode: evt.data.student_code,
          fullName: evt.data.full_name,
          cameraId: evt.data.camera_id,
          cameraName: evt.data.camera_name,
          workerId: evt.data.worker_id,
          timestamp: evt.data.timestamp || new Date().toISOString(),
          similarityScore: evt.data.similarity_score ?? 0,
          livenessScore: evt.data.liveness_score ?? 0,
          verificationStatus: 'CONFIRMED',
        };
        setRecords((prev) => [newRecord, ...prev.filter((r) => r.id !== newRecord.id)]);
      } else {
        fetchAttendance();
      }
    });

    return () => {
      unsubAtt();
    };
  }, [fetchAttendance]);

  return { records, loading, error, refresh: fetchAttendance };
}

export function useStudents(skip: number = 0, limit: number = 100) {
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStudents = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const list = await ApiService.getStudents(skip, limit);
      setStudents(list);
    } catch (err: any) {
      setError(err.message || 'Unable to load student directory');
      setStudents([]);
    } finally {
      setLoading(false);
    }
  }, [skip, limit]);

  useEffect(() => {
    fetchStudents();
  }, [fetchStudents]);

  return { students, setStudents, loading, error, refresh: fetchStudents };
}
