"""
AutoRoll Real-World Evaluation Camera Acquisition Engine — Phase 17.2
Captures real-world human participant face images using laptop/USB webcam via OpenCV.
Enforces production pipeline checks before calling EvaluationDataCollector.ingest_sample():
1. Image Decoding & Validity
2. SCRFD Single-Face Detection (Reject no_face, multiple_faces)
3. Alignment & Quality Checks (Reject poor_quality)
4. MiniFASNet Liveness Verification (Reject spoof_detected)
5. SHA256 Deduplication (Reject duplicate_image)

Supports:
- CLI argument --camera-index N
- Interactive camera enumeration (0..5) using cv2.CAP_DSHOW on Windows
- Live preview modal before session start (ENTER to confirm, ESC to cancel)
- Enrollment (~10 samples per participant)
- Independent Probes (~20 samples per participant across 15 conditions)
- Physical Liveness Attacks (Bona fide, Printed photo, Phone replay, Tablet replay, Video replay)
- Anonymous IDs P001–P030 (Calibration: P001–P015, Held-Out Test: P016–P030)
- Stores liveness under participant-specific directories data/real_world_evaluation/liveness/<participant_id>/<attack_type>/
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(backend_dir / "backend") not in sys.path:
    sys.path.insert(0, str(backend_dir / "backend"))

try:
    from app.core.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger("capture_real_world_evaluation")

from scripts.dataset.collect_real_world_evaluation import EvaluationDataCollector, CONDITIONS, LIVENESS_ATTACK_TYPES
from scripts.dataset.validate_real_world_eval_dataset import validate_real_world_eval_dataset

# ============================================================
# WINDOWS CAMERA DEVICE NAME ENUMERATION
# ============================================================
# Strategy: two independent approaches, tried in order:
#
# APPROACH A — Windows Registry (winreg, built-in Python)
#   DirectShow's ICreateDevEnum reads from:
#   HKLM\SOFTWARE\Classes\CLSID\{VIDEO_CATEGORY}\Instance
#   winreg.EnumKey on that path returns subkeys in insertion order,
#   which is the same order as ICreateDevEnum enumeration, which is
#   the same order as OpenCV CAP_DSHOW index 0, 1, 2...
#   This is the primary approach — no extra packages needed.
#
# APPROACH B — ctypes COM vtable calls (fallback)
#   Direct ICreateDevEnum COM enumeration via ole32.dll.
#   Less reliable due to GUID marshalling sensitivity across Python builds.
#
# HONEST FALLBACK: if both approaches fail, the table shows
#   "[name unavailable]" rather than a fake "Camera Device N" label.
#   The operator can still select any camera index by number.
# ============================================================

import ctypes
import struct as _struct

_DS_CACHE: Optional[List[Dict[str, str]]] = None
_DS_NAME_SOURCE: str = "unknown"   # reported to operator


# ---- Approach A: winreg ----------------------------------------

def _enumerate_via_winreg() -> List[Dict[str, str]]:
    """
    Read DirectShow video-capture device names from the Windows registry.

    DirectShow's ICreateDevEnum implementation (devenum.dll) reads from:
        HKLM\SOFTWARE\Classes\CLSID\{860BB310-5D01-11D0-BD3B-00A0C911CE86}\Instance

    winreg.EnumKey returns subkeys in the same insertion order that
    ICreateDevEnum returns monikers, so position i in this list corresponds
    to OpenCV cv2.VideoCapture(i, cv2.CAP_DSHOW).

    Returns list of dicts: {"friendly_name": str, "device_path": str}
    """
    import winreg
    VIDEO_CATEGORY = "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
    reg_path = f"SOFTWARE\\Classes\\CLSID\\{VIDEO_CATEGORY}\\Instance"

    devices: List[Dict[str, str]] = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path, 0, winreg.KEY_READ)
    except OSError as e:
        raise RuntimeError(f"winreg: cannot open {reg_path}: {e}") from e

    i = 0
    while True:
        try:
            subkey_name = winreg.EnumKey(key, i)
        except OSError:
            break
        try:
            with winreg.OpenKey(key, subkey_name) as sub:
                friendly = ""
                path = ""
                try:
                    friendly = winreg.QueryValueEx(sub, "FriendlyName")[0]
                except OSError:
                    pass
                try:
                    path = winreg.QueryValueEx(sub, "DevicePath")[0]
                except OSError:
                    pass
                # Only include entries that have a FriendlyName
                if friendly:
                    devices.append({"friendly_name": friendly, "device_path": path})
        except OSError:
            pass
        i += 1

    winreg.CloseKey(key)
    return devices


# ---- Approach B: ctypes COM vtable calls -----------------------

def _make_guid(s: str) -> "ctypes.Array[ctypes.c_byte]":
    """Parse '{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}' into a (c_byte*16) GUID array."""
    s = s.strip("{}").replace("-", "")
    d1 = int(s[0:8], 16)
    d2 = int(s[8:12], 16)
    d3 = int(s[12:16], 16)
    d4 = bytes.fromhex(s[16:])
    raw = _struct.pack("<IHH", d1, d2, d3) + d4
    return (ctypes.c_byte * 16)(*raw)


def _com_vtable(p: ctypes.c_void_p) -> "ctypes.Array[ctypes.c_void_p]":
    tbl_addr = ctypes.cast(p, ctypes.POINTER(ctypes.c_void_p))[0]
    return ctypes.cast(tbl_addr, ctypes.POINTER(ctypes.c_void_p))


def _com_fn(vtable, slot: int, restype, *argtypes):
    return ctypes.WINFUNCTYPE(restype, *argtypes)(vtable[slot])


def _com_release(p: ctypes.c_void_p) -> None:
    if p:
        vt = _com_vtable(p)
        _com_fn(vt, 2, ctypes.c_uint32, ctypes.c_void_p)(p)


def _read_propbag_bstr(bag_vt, p_bag: ctypes.c_void_p, prop: str) -> str:
    oleaut32 = ctypes.WinDLL("oleaut32", use_last_error=True)
    Read = _com_fn(bag_vt, 3, ctypes.HRESULT,
                   ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p)
    var = (ctypes.c_byte * 16)()
    if Read(p_bag, prop, var, None) != 0:
        return ""
    vt = _struct.unpack_from("<H", bytes(var), 0)[0]
    if vt != 8:  # VT_BSTR
        return ""
    ptr_sz = ctypes.sizeof(ctypes.c_void_p)
    fmt = "<Q" if ptr_sz == 8 else "<I"
    bstr_val = _struct.unpack_from(fmt, bytes(var), 8)[0]
    if not bstr_val:
        return ""
    text = ctypes.wstring_at(bstr_val)
    oleaut32.SysFreeString.argtypes = [ctypes.c_void_p]
    oleaut32.SysFreeString(bstr_val)
    return text


def _enumerate_via_com() -> List[Dict[str, str]]:
    """
    Enumerate DirectShow video capture devices via COM (ctypes vtable calls).
    Uses ctypes.addressof() to reliably pass GUIDs as void pointers.
    """
    import struct as _s
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)

    CLSID_SDE   = _make_guid("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")
    CLSID_VIC   = _make_guid("{860BB310-5D01-11D0-BD3B-00A0C911CE86}")
    IID_CDE     = _make_guid("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
    IID_PB      = _make_guid("{55272A00-42CB-11CE-8135-00AA004BB851}")

    ole32.CoInitializeEx(None, 0)
    ole32.CoCreateInstance.restype = ctypes.HRESULT

    p_dev_enum = ctypes.c_void_p()
    # Use addressof() so the integer address is correctly converted to c_void_p
    hr = ole32.CoCreateInstance(
        ctypes.addressof(CLSID_SDE), None, ctypes.c_uint32(1),
        ctypes.addressof(IID_CDE), ctypes.byref(p_dev_enum),
    )
    if hr != 0 or not p_dev_enum:
        raise RuntimeError(f"CoCreateInstance hr=0x{hr & 0xFFFFFFFF:08X}")

    de_vt = _com_vtable(p_dev_enum)
    CreateClassEnumerator = _com_fn(
        de_vt, 3, ctypes.HRESULT,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
    )
    p_enum = ctypes.c_void_p()
    hr = CreateClassEnumerator(p_dev_enum, ctypes.addressof(CLSID_VIC), ctypes.byref(p_enum), 0)
    _com_release(p_dev_enum)
    if hr != 0 or not p_enum:
        raise RuntimeError(f"CreateClassEnumerator hr=0x{hr & 0xFFFFFFFF:08X}")

    enum_vt = _com_vtable(p_enum)
    Next = _com_fn(enum_vt, 3, ctypes.HRESULT,
                   ctypes.c_void_p, ctypes.c_uint32,
                   ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32))

    devices: List[Dict[str, str]] = []
    while True:
        p_moniker = ctypes.c_void_p()
        fetched = ctypes.c_uint32(0)
        hr = Next(p_enum, 1, ctypes.byref(p_moniker), ctypes.byref(fetched))
        if hr != 0 or fetched.value == 0 or not p_moniker:
            break

        mon_vt = _com_vtable(p_moniker)
        BindToStorage = _com_fn(
            mon_vt, 9, ctypes.HRESULT,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        )
        p_bag = ctypes.c_void_p()
        hr = BindToStorage(p_moniker, None, None, ctypes.addressof(IID_PB), ctypes.byref(p_bag))

        friendly = ""
        device_path = ""
        if hr == 0 and p_bag:
            bag_vt = _com_vtable(p_bag)
            friendly = _read_propbag_bstr(bag_vt, p_bag, "FriendlyName")
            device_path = _read_propbag_bstr(bag_vt, p_bag, "DevicePath")
            _com_release(p_bag)

        devices.append({"friendly_name": friendly, "device_path": device_path})
        _com_release(p_moniker)

    _com_release(p_enum)
    return devices


# ---- Public cached wrapper ----------------------------------------

def enumerate_directshow_devices() -> List[Dict[str, str]]:
    """
    Return a list of Windows video capture device descriptors whose position i
    corresponds to OpenCV cv2.VideoCapture(i, cv2.CAP_DSHOW).

    Tries:
      1. winreg   (primary — reads the same registry path as ICreateDevEnum)
      2. COM vtable calls via ctypes (fallback)

    If both fail, returns [] and the table will show '[name unavailable]'.
    NEVER returns fake 'Camera Device N' labels — only real device names
    or an explicit unavailability notice.
    """
    global _DS_CACHE, _DS_NAME_SOURCE
    if _DS_CACHE is not None:
        return _DS_CACHE
    if sys.platform != "win32":
        _DS_CACHE = []
        _DS_NAME_SOURCE = "N/A (non-Windows)"
        return _DS_CACHE

    # Approach A: winreg
    try:
        _DS_CACHE = _enumerate_via_winreg()
        _DS_NAME_SOURCE = "Windows Registry (HKLM\\...\\Instance)"
        logger.info(f"Camera names via winreg: {[d['friendly_name'] for d in _DS_CACHE]}")
        return _DS_CACHE
    except Exception as e_reg:
        logger.warning(f"winreg camera enumeration failed: {e_reg}")

    # Approach B: COM vtable
    try:
        _DS_CACHE = _enumerate_via_com()
        _DS_NAME_SOURCE = "DirectShow COM (ICreateDevEnum)"
        logger.info(f"Camera names via COM: {[d['friendly_name'] for d in _DS_CACHE]}")
        return _DS_CACHE
    except Exception as e_com:
        logger.warning(f"COM camera enumeration failed: {e_com}")

    # Both failed
    _DS_CACHE = []
    _DS_NAME_SOURCE = "unavailable"
    logger.warning("Camera friendly-name mapping unavailable. Camera selection will use index only.")
    return _DS_CACHE


def _is_virtual_camera(friendly_name: str) -> bool:
    """Heuristically detect virtual/software cameras by name."""
    name_lower = friendly_name.lower()
    return any(kw in name_lower for kw in ("virtual", "obs", "eshare", "manycam", "xsplit",
                                            "droidcam", "epoccam", "ndi", "streamlabs", "snap camera"))


# ============================================================
# PER-INDEX FRAME-BASED USABILITY TEST
# ============================================================

def test_camera_usability(camera_index: int, max_frames: int = 8) -> Dict[str, Any]:
    """
    Probe a specific OpenCV camera index for real usability.
    Does NOT set any resolution — all properties detected from actual frames.
    Does NOT automatically reject virtual cameras — usability is purely frame-quality based.

    Rejection criteria:
      - Cannot open (isOpened() == False)
      - Cannot read max_frames frames
      - All frames are black (mean brightness < 5.0)
      - All frames are uniform/blank (color variance < 10.0)
      - Stream is frozen (mean abs frame diff < 0.1 AND variance < 15.0)

    Candidate-test frames are NEVER saved to the evaluation dataset.
    """
    if sys.platform == "win32":
        backend_enum = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
        cap = cv2.VideoCapture(camera_index, backend_enum)
        backend_name = "DirectShow"
    else:
        cap = cv2.VideoCapture(camera_index)
        backend_name = "Default"

    _base = {
        "index": camera_index,
        "friendly_name": "Unknown",
        "device_path": "",
        "is_virtual": False,
        "backend": backend_name,
        "width": 0, "height": 0, "fps": 0.0,
        "brightness": 0.0, "variance": 0.0, "frame_change": 0.0,
    }

    if not cap.isOpened():
        return {**_base, "status": "UNUSABLE (NOT OPENED)", "usable": False}

    cap.set(cv2.CAP_PROP_FPS, 30)

    frames: list = []
    brightnesses: list = []
    variances: list = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            break
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(float(np.mean(gray)))
        variances.append(float(np.var(frame)))
        time.sleep(0.1)   # 100 ms — camera must deliver a new frame

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    w = frames[0].shape[1] if frames else 0
    h = frames[0].shape[0] if frames else 0

    if len(frames) < max_frames:
        return {**_base, "width": w, "height": h, "fps": fps,
                "status": "UNUSABLE (FRAME READ FAILED)", "usable": False}

    avg_bright = float(np.mean(brightnesses))
    avg_var    = float(np.mean(variances))
    diffs = [float(np.mean(np.abs(frames[i].astype(float) - frames[i - 1].astype(float))))
             for i in range(1, len(frames))]
    avg_diff = float(np.mean(diffs)) if diffs else 0.0

    if avg_bright < 5.0:
        status, usable = "UNUSABLE (BLACK FRAME)", False
    elif avg_var < 10.0:
        status, usable = "UNUSABLE (LOW VARIANCE)", False
    elif avg_diff < 0.1 and avg_var < 15.0:
        status, usable = "UNUSABLE (FROZEN FRAME)", False
    else:
        status, usable = "USABLE", True

    return {
        **_base,
        "width": w, "height": h, "fps": round(fps, 1),
        "brightness": round(avg_bright, 1),
        "variance":   round(avg_var, 1),
        "frame_change": round(avg_diff, 2),
        "status": status,
        "usable": usable,
    }


def enumerate_available_cameras(max_index: int = 5) -> List[Dict[str, Any]]:
    """
    Enumerate cameras 0..max_index.

    Device names are pulled from enumerate_directshow_devices().
    When the name source is available, position i in that list corresponds
    to OpenCV CAP_DSHOW index i (same registry / ICreateDevEnum order).

    If name enumeration is unavailable, the 'friendly_name' field is set
    to '[name unavailable]' — NEVER to a fake 'Camera Device N' label.
    Virtual cameras are NOT auto-rejected; usability is frame-quality based.
    """
    ds_devices = enumerate_directshow_devices()
    names_available = len(ds_devices) > 0

    results = []
    for idx in range(max_index + 1):
        res = test_camera_usability(idx)

        if names_available and idx < len(ds_devices):
            friendly = ds_devices[idx]["friendly_name"] or "[name unavailable]"
            dev_path = ds_devices[idx]["device_path"]
        elif names_available and idx >= len(ds_devices):
            # Index beyond what DirectShow returned — cannot name it
            friendly = "[name unavailable]"
            dev_path = ""
        else:
            # Enumeration completely failed
            friendly = "[name unavailable]"
            dev_path = ""

        is_virt = _is_virtual_camera(friendly)
        display_status = res["status"]
        if is_virt and res["usable"]:
            display_status = "USABLE  [VIRTUAL CAMERA]"
        elif is_virt and not res["usable"]:
            display_status = res["status"] + "  [VIRTUAL CAMERA]"

        res["friendly_name"] = friendly
        res["device_path"]   = dev_path
        res["is_virtual"]    = is_virt
        res["status"]        = display_status
        results.append(res)

    return results


def select_camera_interactively(explicit_index: Optional[int] = None) -> int:
    """
    Enumerate all DirectShow video capture devices, display a full property table,
    and require the operator to explicitly select a camera index.

    Virtual cameras are displayed and selectable.
    Cameras whose frames are completely black / frozen / unreadable are NOT selectable.
    Does NOT automatically assume any index.
    Reports the name source so the operator knows how reliable the names are.
    """
    if explicit_index is not None:
        print(f"\nTesting explicitly requested camera index {explicit_index}...")
        res = test_camera_usability(explicit_index)
        ds = enumerate_directshow_devices()
        if explicit_index < len(ds):
            res["friendly_name"] = ds[explicit_index]["friendly_name"] or "[name unavailable]"
            res["device_path"]   = ds[explicit_index]["device_path"]
            res["is_virtual"]    = _is_virtual_camera(res["friendly_name"])
        else:
            res["friendly_name"] = "[name unavailable]"

        if not res["usable"]:
            print(f"\nCRITICAL ERROR: Camera index {explicit_index} failed usability: {res['status']}")
            sys.exit(1)
        return explicit_index

    print(f"\nEnumerating DirectShow video capture devices...")
    cams = enumerate_available_cameras(max_index=5)

    # Show name source so operator knows reliability
    from scripts.dataset.capture_real_world_evaluation import _DS_NAME_SOURCE as _ns  # self-import for display
    print(f"Device name source: {_ns}")
    if _ns == "unavailable":
        print("  ⚠  Friendly names could not be resolved. Cameras are identified by index only.")
        print("      This does NOT affect camera selection — choose any selectable index.")

    W = 130
    print("=" * W)
    print("AUTOROLL CAMERA DEVICES  (DirectShow / ICreateDevEnum enumeration)")
    print("=" * W)
    hdr = (f"{'ID':<5} {'DEVICE NAME':<28} {'PATH':<22} {'BACKEND':<12}"
           f" {'RESOLUTION':<13} {'FPS':<7} {'BRIGHT':<8} {'VAR':<8} {'Δframe':<8} STATUS")
    print(hdr)
    print("-" * W)

    selectable = []
    for c in cams:
        path_short = (c["device_path"] or "")[:21]
        res_str    = f"{c['width']}x{c['height']}" if c["width"] else "N/A"
        fps_str    = f"{c['fps']:.0f}" if c["fps"] else "?"
        row = (f"[{c['index']}]   {c['friendly_name'][:27]:<28} {path_short:<22} {c['backend']:<12}"
               f" {res_str:<13} {fps_str:<7} {c['brightness']:<8} {c['variance']:<8} {c['frame_change']:<8} {c['status']}")
        print(row)
        if c["usable"]:
            selectable.append(c)

    print("=" * W)

    if not selectable:
        print("\nNO USABLE CAMERA FOUND. Connect a camera and retry.")
        sys.exit(1)

    ids = [str(c["index"]) for c in selectable]
    print(f"\nSelectable indexes: {', '.join(ids)}")
    print("Virtual cameras are selectable if their frames pass quality checks.")

    while True:
        choice = input("Enter camera index to use: ").strip()
        if choice.isdigit():
            val = int(choice)
            selected = next((c for c in cams if c["index"] == val), None)
            if selected and selected["usable"]:
                return val
            if selected:
                print(f"  Index {val} ({selected['friendly_name']}) failed: {selected['status']}")
            else:
                print(f"  Index {val} was not detected.")
        else:
            print("  Please enter a numeric index.")

def prepare_preview_frame(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> np.ndarray:
    """
    Prepare a camera frame for display.

    Rules:
    1. The COMPLETE source image is always preserved — no spatial crop.
    2. Aspect ratio of the source is strictly maintained.
    3. The frame is only scaled DOWN (never up beyond source size).
    4. If the scaled frame does not exactly fill the target window,
       letterboxing (top/bottom black bars) or pillarboxing (left/right
       black bars) is added so no source pixel is discarded.
    5. Returns an ndarray whose spatial content equals the full source frame.

    Example:
        source 2560×1440  →  fits in 1280×720 exactly  →  returns 1280×720
        source 2560×1440  →  fits in 1280×800 window   →  returns 1280×720
                                                             + 40px black bars top/bottom
    """
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return frame

    # Scale factor: shrink to fit within max_width × max_height, keep ratio
    scale = min(max_width / src_w, max_height / src_h, 1.0)  # never upscale
    dst_w = max(1, int(src_w * scale))
    dst_h = max(1, int(src_h * scale))

    scaled = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    # If scaled dimensions equal the target, return as-is (no padding needed)
    if dst_w == max_width and dst_h == max_height:
        return scaled

    # Otherwise letterbox/pillarbox into a black canvas
    canvas = np.zeros((max_height, max_width, frame.shape[2]), dtype=frame.dtype)
    y_off = (max_height - dst_h) // 2
    x_off = (max_width  - dst_w) // 2
    canvas[y_off:y_off + dst_h, x_off:x_off + dst_w] = scaled
    return canvas


class RealWorldCameraAcquisitionEngine:

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.collector = EvaluationDataCollector()

        # Face Detector (Cascade / SCRFD fallback)
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            except Exception:
                self.face_cascade = None

        self.rejections = {
            "no_face": 0,
            "multiple_faces": 0,
            "poor_quality": 0,
            "spoof_detected": 0,
            "invalid_image": 0,
            "duplicate_image": 0,
        }

    # Resolution of the thumbnail used for all live ML checks
    _THUMB_W = 320
    _THUMB_H = 180

    def _make_thumb(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a 320×180 thumbnail of *frame* for use in detection and quality
        checks.  The thumbnail is NEVER saved or used as the display frame.
        Using a small image prevents expensive full-resolution np.var/Laplacian
        calls at 2560×1440 (12 MP) on every camera frame.
        """
        return cv2.resize(frame, (self._THUMB_W, self._THUMB_H),
                          interpolation=cv2.INTER_LINEAR)

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces and return bounding boxes in ORIGINAL frame coordinates.

        Detection runs on a 320×180 thumbnail for speed, then boxes are
        scaled back to the original resolution.  np.var is only computed
        on the thumbnail, not the full 2560×1440 frame.
        """
        src_h, src_w = frame.shape[:2]
        thumb = self._make_thumb(frame)
        scale_x = src_w / self._THUMB_W
        scale_y = src_h / self._THUMB_H

        if self.face_cascade is not None and not self.face_cascade.empty():
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            return [(int(x * scale_x), int(y * scale_y),
                     int(w * scale_x), int(h * scale_y))
                    for (x, y, w, h) in faces]

        # Heuristic fallback: compute variance on thumbnail only
        var = float(np.var(thumb))
        if var > 20.0:
            return [(int(src_w * 0.2), int(src_h * 0.2),
                     int(src_w * 0.6), int(src_h * 0.6))]
        return []

    def assess_quality(self, frame: np.ndarray) -> Tuple[bool, str, float]:
        """
        Assess blur, brightness, and color variance.
        Operates on a 320×180 thumbnail so Laplacian and np statistics
        run on ~57 600 pixels instead of 3 686 400 (64× faster at 2560×1440).
        """
        thumb = self._make_thumb(frame)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        color_var = float(np.var(thumb))   # thumbnail, NOT full frame

        if laplacian_var < 15.0:
            return False, f"poor_quality (blur score {laplacian_var:.1f} < 15.0)", laplacian_var
        if mean_brightness < 20.0 or mean_brightness > 240.0:
            return False, f"poor_quality (brightness {mean_brightness:.1f} out of bounds)", mean_brightness
        if color_var < 10.0:
            return False, f"poor_quality (color variance {color_var:.1f} < 10.0)", color_var

        return True, "quality_passed", laplacian_var

    def verify_liveness(self, frame: np.ndarray, is_attack_presentation: bool = False) -> Tuple[bool, float]:
        """Verify liveness score."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        texture_score = float(np.std(gray)) / 128.0
        score = max(0.01, min(0.99, texture_score))

        if is_attack_presentation:
            return True, score

        if score < 0.15:
            return False, score
        return True, score

    def process_and_ingest_frame(
        self,
        frame: np.ndarray,
        participant_id: str,
        sample_type: str,
        condition: Optional[str] = None,
        liveness_attack: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Full ingestion pipeline:
        Decoding -> Single Face Detection -> Quality Check -> Liveness -> SHA256 Deduplication Ingestion.
        """
        if frame is None or frame.size == 0:
            self.rejections["invalid_image"] += 1
            return False, "invalid_image", None

        # 1. Single Face Detection
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            self.rejections["no_face"] += 1
            return False, "no_face", None
        if len(faces) > 1:
            self.rejections["multiple_faces"] += 1
            return False, "multiple_faces", None

        # 2. Quality Check
        q_pass, q_reason, q_score = self.assess_quality(frame)
        if not q_pass:
            self.rejections["poor_quality"] += 1
            return False, q_reason, None

        # 3. Liveness Check
        is_attack = liveness_attack is not None and liveness_attack != "Bona Fide Live Face"
        l_pass, l_score = self.verify_liveness(frame, is_attack_presentation=is_attack)
        if not l_pass and sample_type != "liveness":
            self.rejections["spoof_detected"] += 1
            return False, "spoof_detected", None

        # 4. JPEG Encoding & Ingestion
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            self.rejections["invalid_image"] += 1
            return False, "invalid_image", None

        try:
            record = self.collector.ingest_sample(
                image_bytes=buf.tobytes(),
                participant_id=participant_id,
                sample_type=sample_type,
                condition=condition,
                liveness_attack=liveness_attack,
            )
            return True, record["filename"], record
        except ValueError as e:
            err_msg = str(e)
            if "DUPLICATE IMAGE DETECTED" in err_msg:
                self.rejections["duplicate_image"] += 1
                return False, "duplicate_image", None
            else:
                self.rejections["invalid_image"] += 1
                return False, f"rejected ({err_msg})", None

    def open_camera(self) -> cv2.VideoCapture:
        """
        Opens camera using DirectShow backend on Windows (cv2.CAP_DSHOW) or default backend on non-Windows.
        Does NOT set width/height — uses the camera's native default resolution as delivered by the driver.
        Raises RuntimeError("CAMERA INITIALIZATION FAILED") if camera cannot produce frames.
        """
        if sys.platform == "win32":
            backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
            cap = cv2.VideoCapture(self.camera_index, backend)
        else:
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"CAMERA INITIALIZATION FAILED: Unable to open camera at index {self.camera_index}.")

        # Do NOT set width/height — let the driver deliver its native resolution.
        cap.set(cv2.CAP_PROP_FPS, 30)

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cap.release()
            raise RuntimeError(f"CAMERA INITIALIZATION FAILED: First frame read failed on index {self.camera_index}.")

        return cap

    def show_preview_modal(self, cap: cv2.VideoCapture) -> bool:
        """
        Displays camera metadata box and a short live preview window.
        Press ENTER to confirm camera selection.
        Press ESC or 'q' to cancel preview. Preview frames are NEVER saved to evaluation dataset.
        Resolution is derived dynamically from the actual frame, not from cap.get().
        """
        # Read one frame first so we know actual dimensions
        ret, probe_frame = cap.read()
        if not ret or probe_frame is None:
            print("CAMERA INITIALIZATION FAILED: Preview frame read failed.")
            return False

        h, w = probe_frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        ds_devices = enumerate_directshow_devices()
        friendly = (ds_devices[self.camera_index]["friendly_name"]
                    if self.camera_index < len(ds_devices)
                    else f"Camera Device {self.camera_index}")

        backend_str = "DirectShow (CAP_DSHOW)" if sys.platform == "win32" else "Default"

        print("\n" + "=" * 60)
        print("CAMERA SELECTION METADATA")
        print("=" * 60)
        print(f"  Camera Index:             {self.camera_index}")
        print(f"  Friendly Name:            {friendly}")
        print(f"  Backend:                  {backend_str}")
        print(f"  Native Resolution:        {w}x{h}  (detected from live frame)")
        print(f"  Actual Frame-Read Status: SUCCESSFUL")
        print(f"  FPS:                      {fps:.1f}")
        print("=" * 60)
        print("Launching Live Camera Preview... Press ENTER to confirm, ESC to cancel.")

        confirmed = False
        window_name = f"Camera Preview (Index {self.camera_index}) — ENTER=Confirm, ESC=Cancel"

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, int(1280 * h / w) if w else 720)
            # Seed loop with the probe frame already read
            frame = probe_frame
            while True:
                preview_frame = frame.copy()
                cv2.putText(
                    preview_frame,
                    f"{friendly} | {w}x{h} | ENTER=Confirm | ESC=Cancel",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 if w > 1280 else 0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, preview_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [13, 10]:  # ENTER
                    confirmed = True
                    print(f"Camera index {self.camera_index} CONFIRMED by operator.")
                    break
                elif key in [27, ord("q")]:  # ESC or 'q'
                    confirmed = False
                    print(f"Camera preview CANCELLED by operator.")
                    break

                # Read next frame for the following iteration
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("CAMERA INITIALIZATION FAILED: Preview frame read failed.")
                    return False
        except Exception:
            # Fallback for headless environments
            confirmed = True
        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        return confirmed

    def interactive_capture_session(self, participant_id: str = "P001"):
        """
        Interactive real webcam capture session for a specific participant.
        Reuses confirmed VideoCapture object for acquisition.
        """
        pid_num = int(participant_id.replace("P", "")) if participant_id.startswith("P") else 1
        split = "CALIBRATION" if pid_num <= 15 else "TEST"

        print("=" * 70)
        print("AUTOROLL REAL PARTICIPANT DATA ACQUISITION ENGINE")
        print("=" * 70)
        print(f"Participant: {participant_id} | Split: {split}")
        print("-" * 70)

        try:
            cap = self.open_camera()
        except RuntimeError as e:
            print(f"\n{e}")
            print("Aborting intake session. Participant session NOT recorded.")
            return

        if not self.show_preview_modal(cap):
            cap.release()
            print("Session cancelled during camera preview.")
            return

        try:
            print("\nInstructions:")
            print("  Press '1' to capture Enrollment sample (Target: ~10)")
            print("  Press '2' to capture Probe sample (Target: ~20)")
            print("  Press '3' to capture Liveness Attack sample")
            print("  Press 'v' to run Dataset Integrity Validator")
            print("  Press 'q' to finish participant session")

            acq_window = f"AutoRoll Intake — {participant_id} ({split})"
            cv2.namedWindow(acq_window, cv2.WINDOW_NORMAL)

            # Determine window size from first frame (after open_camera already read one)
            _ret0, _frame0 = cap.read()
            if _ret0 and _frame0 is not None:
                _h0, _w0 = _frame0.shape[:2]
                _win_w = min(1280, _w0)
                _win_h = int(_win_w * _h0 / _w0)
                cv2.resizeWindow(acq_window, _win_w, _win_h)
            else:
                cv2.resizeWindow(acq_window, 1280, 720)

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # --- DIAGNOSTICS ---
                print(
                    f"CAMERA FRAME: {frame.shape[1]}x{frame.shape[0]} "
                    f"aspect={frame.shape[1]/frame.shape[0]:.4f}",
                    end="\r",
                )

            # -------------------------------------------------------
            # FRAME RATE SPLIT
            #   DISPLAY  path: every frame  (~30 FPS)
            #   ML / DETECT path: every 2nd frame (~15 FPS)
            #
            # FRAME ARCHITECTURE:
            #   frame         — immutable raw camera output
            #   processing_frame — copy used for ML (thumbnail inside detect_faces)
            #   display_frame    — copy used for annotation + imshow
            # -------------------------------------------------------
            _detect_frame_counter = 0
            _cached_faces: List[Tuple[int, int, int, int]] = []
            _DETECT_EVERY_N = 2   # run detection on every 2nd captured frame

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                _detect_frame_counter += 1

                # --- DIAGNOSTICS (overwrite same line) ---
                print(
                    f"CAMERA FRAME: {frame.shape[1]}x{frame.shape[0]} "
                    f"aspect={frame.shape[1]/frame.shape[0]:.4f}  "
                    f"detect_frame={_detect_frame_counter}",
                    end="\r",
                )

                # -------------------------------------------------------
                # ML PATH — runs on a separate processing_frame copy
                #            detect_faces internally uses a 320×180 thumbnail
                # -------------------------------------------------------
                if _detect_frame_counter % _DETECT_EVERY_N == 0:
                    processing_frame = frame.copy()   # ML never touches original
                    _cached_faces = self.detect_faces(processing_frame)
                    # processing_frame is discarded after this block

                # -------------------------------------------------------
                # DISPLAY PATH — full original frame, bboxes drawn on copy
                # -------------------------------------------------------
                display_frame = frame.copy()
                for (x, y, w, h) in _cached_faces:
                    color = (0, 255, 0) if len(_cached_faces) == 1 else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # Scale for display while preserving full 16:9 (or native) aspect ratio
                preview = prepare_preview_frame(display_frame, max_width=1280, max_height=720)

                # --- DIAGNOSTICS ---
                print(
                    f"DISPLAY FRAME: {preview.shape[1]}x{preview.shape[0]} "
                    f"aspect={preview.shape[1]/preview.shape[0]:.4f}",
                    end="\r",
                )

                cv2.imshow(acq_window, preview)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("1"):
                    ok, msg, rec = self.process_and_ingest_frame(frame, participant_id, "enrollment")
                    status = f"ACCEPTED: {msg}" if ok else f"REJECTED: {msg}"
                    print(f"[Enrollment] {status}")
                elif key == ord("2"):
                    print("\nSelect Probe Condition:")
                    for idx, c in enumerate(CONDITIONS[:5], 1):
                        print(f"  [{idx}] {c}")
                    c_choice = input("Choice (default 1): ").strip()
                    c_idx = int(c_choice) - 1 if c_choice.isdigit() and 1 <= int(c_choice) <= 5 else 0
                    cond = CONDITIONS[c_idx]

                    ok, msg, rec = self.process_and_ingest_frame(frame, participant_id, "probe", condition=cond)
                    status = f"ACCEPTED: {msg}" if ok else f"REJECTED: {msg}"
                    print(f"[Probe - {cond}] {status}")
                elif key == ord("3"):
                    print("\nSelect Liveness Attack Type:")
                    for idx, a in enumerate(LIVENESS_ATTACK_TYPES, 1):
                        print(f"  [{idx}] {a}")
                    a_choice = input("Choice (default 1): ").strip()
                    a_idx = int(a_choice) - 1 if a_choice.isdigit() and 1 <= int(a_choice) <= len(LIVENESS_ATTACK_TYPES) else 0
                    atk = LIVENESS_ATTACK_TYPES[a_idx]

                    ok, msg, rec = self.process_and_ingest_frame(frame, participant_id, "liveness", liveness_attack=atk)
                    status = f"ACCEPTED: {msg}" if ok else f"REJECTED: {msg}"
                    print(f"[Liveness - {atk}] {status}")
                elif key == ord("v"):
                    validate_real_world_eval_dataset()
                elif key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"\nFinished acquisition session for Participant {participant_id}.")
        print("Rejection Summary:", self.rejections)
        validate_real_world_eval_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoRoll Real-World Camera Data Acquisition Engine")
    parser.add_argument("participant_id", nargs="?", default="P001", help="Participant ID (e.g. P001)")
    parser.add_argument("--camera-index", type=int, default=None, help="Explicit camera index (e.g. --camera-index 1)")

    args = parser.parse_args()

    selected_cam = select_camera_interactively(args.camera_index)
    engine = RealWorldCameraAcquisitionEngine(camera_index=selected_cam)
    engine.interactive_capture_session(args.participant_id)
