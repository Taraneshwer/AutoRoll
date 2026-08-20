"""
Camera Repository for database operations.
"""


from sqlalchemy.orm import Session

from server.app.db.models import Camera


class CameraRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, name: str, rtsp_url: str, location: str | None = None) -> Camera:
        cam = Camera(name=name, rtsp_url=rtsp_url, location=location)
        self.db.add(cam)
        self.db.commit()
        self.db.refresh(cam)
        return cam

    def get_by_id(self, camera_id: str) -> Camera | None:
        return self.db.query(Camera).filter(Camera.id == camera_id).first()

    def list_all(self) -> list[Camera]:
        return self.db.query(Camera).all()

    def assign_worker(self, camera_id: str, worker_id: str | None) -> Camera | None:
        cam = self.get_by_id(camera_id)
        if cam:
            cam.assigned_worker_id = worker_id
            self.db.commit()
            self.db.refresh(cam)
        return cam
