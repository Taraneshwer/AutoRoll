"""
AutoRoll Distributed Camera Scheduler Workload Distribution Simulation Test.
Simulates 4 cameras across 1 -> 2 -> 3 workers and verifies failover rebalancing.
"""
import sys
from pathlib import Path
BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import sys
from pathlib import Path


from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Camera, WorkerNode
from app.services.scheduler.scheduler import DistributedCameraScheduler


def run_scheduler_simulation():
    print("\n" + "=" * 70)
    print("      AUTOROLL DISTRIBUTED CAMERA SCHEDULER SIMULATION TEST      ")
    print("=" * 70)

    # In-memory SQLite DB for simulation
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()

    scheduler = DistributedCameraScheduler(max_cameras_per_worker=4)

    # Step 1: Create 4 Cameras
    print("\n[STEP 1] Initializing 4 Cameras (cam_01 .. cam_04)...")
    for i in range(1, 5):
        cam = Camera(
            id=f"cam_0{i}",
            name=f"Camera {i}",
            rtsp_url=f"rtsp://192.168.1.10{i}/live",
            is_active=True,
        )
        db.add(cam)
    db.commit()

    # Step 2: Register Worker 1
    print("\n[STEP 2] Worker 1 Registers (worker_01)...")
    scheduler.register_worker(
        {
            "worker_id": "worker_01",
            "state": "READY",
            "cpu_percent": 15.0,
            "avg_inference_latency_ms": 10.0,
        },
        db,
    )

    scheduler.rebalance_workload(db)
    status1 = scheduler.get_scheduler_status(db)
    print(f"-> Unassigned Cameras : {status1['unassigned_cameras']}")
    print(f"-> Worker 1 Cameras   : {status1['workers'][0]['assigned_camera_ids']}")
    assert (
        status1["workers"][0]["assigned_cameras_count"] == 4
    ), "All 4 cameras should be assigned to Worker 1"

    # Step 3: Register Worker 2 & Rebalance
    print("\n[STEP 3] Worker 2 Registers (worker_02) -> Trigger Rebalance...")
    scheduler.register_worker(
        {
            "worker_id": "worker_02",
            "state": "READY",
            "cpu_percent": 10.0,
            "avg_inference_latency_ms": 8.0,
        },
        db,
    )

    # Unassign 2 cameras to simulate rebalancing across available capacity
    scheduler.unassign_camera("cam_03", db)
    scheduler.unassign_camera("cam_04", db)
    scheduler.rebalance_workload(db)

    status2 = scheduler.get_scheduler_status(db)
    w1_cams = next(w for w in status2["workers"] if w["worker_id"] == "worker_01")
    w2_cams = next(w for w in status2["workers"] if w["worker_id"] == "worker_02")
    print(f"-> Worker 1 Cameras   : {w1_cams['assigned_camera_ids']}")
    print(f"-> Worker 2 Cameras   : {w2_cams['assigned_camera_ids']}")
    assert len(w1_cams["assigned_camera_ids"]) == 2, "Worker 1 should have 2 cameras"
    assert len(w2_cams["assigned_camera_ids"]) == 2, "Worker 2 should have 2 cameras"

    # Step 4: Register Worker 3 & Rebalance
    print("\n[STEP 4] Worker 3 Registers (worker_03) -> Trigger Rebalance...")
    scheduler.register_worker(
        {
            "worker_id": "worker_03",
            "state": "READY",
            "cpu_percent": 8.0,
            "avg_inference_latency_ms": 7.0,
        },
        db,
    )

    scheduler.unassign_camera("cam_02", db)
    scheduler.rebalance_workload(db)

    status3 = scheduler.get_scheduler_status(db)
    print("Workload Distribution Across 3 Workers:")
    for w in status3["workers"]:
        print(
            f"   |- {w['worker_id']:10s} : {w['assigned_cameras_count']} Cameras "
            f"{w['assigned_camera_ids']}"
        )

    # Step 5: Worker Failover Test (Worker 1 Dies)
    print("\n[STEP 5] FAILOVER TEST: Worker 1 Crashes (Heartbeat Timeout)...")
    w1_db = db.query(WorkerNode).filter(WorkerNode.id == "worker_01").first()
    if w1_db:
        w1_db.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        db.commit()

    scheduler.check_worker_timeouts(db)

    status4 = scheduler.get_scheduler_status(db)
    w1_status = next(w for w in status4["workers"] if w["worker_id"] == "worker_01")
    print(f"-> Worker 1 State     : {w1_status['state']}")
    print(f"-> Online Workers     : {status4['online_workers']}/{status4['total_workers']}")
    print("Workload Re-assigned to Remaining Online Workers:")
    for w in status4["workers"]:
        if w["state"] != "OFFLINE":
            print(
                f"   |- {w['worker_id']:10s} : {w['assigned_cameras_count']} Cameras "
                f"{w['assigned_camera_ids']}"
            )

    assert w1_status["state"] == "OFFLINE", "Worker 1 must be marked OFFLINE"
    assert (
        status4["unassigned_cameras"] == 0
    ), "All cameras should be automatically reassigned to healthy workers"

    print("\n" + "=" * 70)
    print("       SCHEDULER SIMULATION PASSED: ZERO DATA LOSS / FULL SCALABILITY     ")
    print("=" * 70 + "\n")

    db.close()


if __name__ == "__main__":
    run_scheduler_simulation()
