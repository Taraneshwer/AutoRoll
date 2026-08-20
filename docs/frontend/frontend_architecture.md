# Frontend Architecture Guide — AutoRoll Phase 13

## 1. Executive Summary

The AutoRoll frontend is a modern React application built with TypeScript, Vite, Tailwind CSS, and Lucide Icons. It connects directly to the AutoRoll FastAPI backend and WebSocket event bus (`/ws/events` and `/ws/monitoring`).

```
                              AutoRoll Frontend Architecture
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
        REST API Layer               WebSocket Engine            Design Tokens
      (ApiService / Client)        (useWebSocket Hook)      (Turkish Blue #007A99, White)
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                                  Page Views & Router
      ┌──────────────────┬──────────────────┼──────────────────┬──────────────────┐
      ▼                  ▼                  ▼                  ▼                  ▼
  Dashboard        Live Monitor         Enrollment          Students          Attendance
```

---

## 2. Visual Identity & Design Tokens

- **Canvas Background:** White / Slate Light (`#FFFFFF` / `#F8FAFC`).
- **Typography:** Charcoal & Black (`#0F172A`).
- **Accent Color:** **Turkish Blue** (`#007A99` / `#0099B8`).
- **Accent Placement:** Navigation buttons, active tab indicators, metric badges, focus rings, action controls.
- **Card Design System:** White card background with subtle borders (`#E2E8F0`), generous 20px padding, subtle shadow transitions on hover.

---

## 3. Page Views Overview

1. **Dashboard (`DashboardPage.tsx`):** Real-time cluster metrics, active cameras, online workers, attendance today, recognition FPS, and P95 latency. Displays prominent `"BACKEND OFFLINE"` alert banner if server is unreachable.
2. **Live Monitor (`LiveMonitoringPage.tsx`):** Low-latency MJPEG video stream feed with real-time detection bounding box overlays, candidate identity, liveness score, FPS, and per-stage latency breakdown.
3. **Enrollment Wizard (`EnrollmentPage.tsx`):** Guided 6-Step enrollment flow: Student Info $\rightarrow$ Camera Access $\rightarrow$ Face Target Box $\rightarrow$ Sample Collection (1/10 .. 10/10) $\rightarrow$ Quality & Liveness Verification $\rightarrow$ Normalized Mean Template Aggregation & Completion.
4. **Students (`StudentsPage.tsx`):** Enrolled student directory with search filtering and biometric template deletion modal warning (`"This will permanently delete the student's biometric template."`).
5. **Attendance (`AttendancePage.tsx`):** Real-time attendance log audit stream with student, camera, worker, similarity, liveness, and status filters.
6. **Cameras (`CamerasPage.tsx`):** RTSP camera configuration and worker node assignment controls.
7. **Workers (`WorkersPage.tsx`):** ML worker cluster telemetry, GPU utilization, VRAM memory, and active stream load scores.
8. **Settings (`SettingsPage.tsx`):** Clear distinction between server-controlled parameters (`AUTOROLL_RECOGNITION_THRESHOLD=0.0540`, `AUTOROLL_ATTENDANCE_COOLDOWN_SECONDS=30`) and client settings.
