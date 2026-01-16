import json
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import cv2

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

import os
import glob
from pathlib import Path

# ----------------------------
# Math: similarity transform (Procrustes)
# ----------------------------
def similarity_transform(target_xy: np.ndarray, moving_xy: np.ndarray, allow_scale: bool = True):
    """
    Compute similarity transform mapping moving_xy -> target_xy.
    Returns R (2x2), s (scalar), t (2,)
    """
    X = np.asarray(target_xy, dtype=float)
    Y = np.asarray(moving_xy, dtype=float)

    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Point arrays must have shape (N,2) and match.")

    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 point pairs for similarity transform.")

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0 = X - muX
    Y0 = Y - muY

    C = (Y0.T @ X0) / n
    U, S, Vt = np.linalg.svd(C)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    if allow_scale:
        varY = (Y0 ** 2).sum() / n
        s = (S.sum() / varY) if varY > 0 else 1.0
    else:
        s = 1.0

    # Row-vector convention: p' = s*(p @ R) + t
    t = muX - s * (muY @ R)
    return R, s, t


def affine_2x3_from(R: np.ndarray, s: float, t: np.ndarray) -> np.ndarray:
    """
    Convert row-vector similarity to OpenCV warpAffine 2x3 (column-vector form).
    """
    A = s * R  # 2x2
    M = np.zeros((2, 3), dtype=float)
    M[:, :2] = A.T
    M[:, 2] = t
    return M


def apply_affine_2x3(M: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    P = np.hstack([pts, ones])  # (N,3) row vectors
    return (P @ M.T)


# ----------------------------
# Image helpers
# ----------------------------
def to_grayscale_float(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")
    if img.ndim == 2:
        out = img
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        tmp = img
        if tmp.dtype != np.uint8:
            tmp = tmp.astype(np.float32)
            tmp = tmp - tmp.min()
            mx = tmp.max()
            if mx > 0:
                tmp = tmp / mx
            tmp = (tmp * 255).clip(0, 255).astype(np.uint8)
        if tmp.shape[2] == 4:
            tmp = tmp[:, :, :3]
        out = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return out.astype(np.float32)


@dataclass
class PreprocessBSettings:
    mode: str = "match_A"   # "none" | "match_A" | "custom" | "scale"
    custom_w: int = 2048
    custom_h: int = 2048
    scale: float = 1.0
    keep_aspect: bool = True
    aspect_strategy: str = "pad"  # "pad" | "crop" | "stretch"
    interpolation: str = "area"   # "nearest" | "linear" | "cubic" | "lanczos" | "area"


def cv_interp(name: str) -> int:
    name = (name or "").lower()
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }.get(name, cv2.INTER_AREA)


def resize_with_aspect(img: np.ndarray, target_wh: Tuple[int, int], keep_aspect: bool,
                       strategy: str, interp: int) -> np.ndarray:
    tgt_w, tgt_h = int(target_wh[0]), int(target_wh[1])
    if tgt_w <= 0 or tgt_h <= 0:
        raise ValueError("Target size must be positive.")

    h, w = img.shape[:2]
    if (not keep_aspect) or strategy == "stretch":
        return cv2.resize(img, (tgt_w, tgt_h), interpolation=interp)

    scale_fit = min(tgt_w / w, tgt_h / h)
    scale_fill = max(tgt_w / w, tgt_h / h)

    if strategy == "pad":
        s = scale_fit
    elif strategy == "crop":
        s = scale_fill
    else:
        s = scale_fit

    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if strategy == "pad":
        out = np.zeros((tgt_h, tgt_w), dtype=resized.dtype)
        y0 = (tgt_h - new_h) // 2
        x0 = (tgt_w - new_w) // 2
        out[y0:y0 + new_h, x0:x0 + new_w] = resized
        return out

    # crop
    y0 = max(0, (new_h - tgt_h) // 2)
    x0 = max(0, (new_w - tgt_w) // 2)
    return resized[y0:y0 + tgt_h, x0:x0 + tgt_w]


def preprocess_B(imgB: np.ndarray, imgA: Optional[np.ndarray], s: PreprocessBSettings) -> np.ndarray:
    interp = cv_interp(s.interpolation)
    if s.mode == "none" or imgB is None:
        return imgB

    if s.mode == "match_A":
        if imgA is None:
            return imgB
        H, W = imgA.shape[:2]
        return resize_with_aspect(imgB, (W, H), s.keep_aspect, s.aspect_strategy, interp)

    if s.mode == "custom":
        return resize_with_aspect(imgB, (s.custom_w, s.custom_h), s.keep_aspect, s.aspect_strategy, interp)

    if s.mode == "scale":
        h, w = imgB.shape[:2]
        new_w = max(1, int(round(w * float(s.scale))))
        new_h = max(1, int(round(h * float(s.scale))))
        return cv2.resize(imgB, (new_w, new_h), interpolation=interp)

    return imgB


def batch_apply_images(
    in_folder: str,
    out_folder: str,
    patterns: List[str],
    imgA: np.ndarray,
    settingsB: PreprocessBSettings,
    M: np.ndarray,
    overwrite: bool,
    log_cb=None,
):
    in_folder = str(in_folder)
    out_folder = str(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    H, W = imgA.shape[:2]

    # Expand patterns
    files = []
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        files.extend(glob.glob(os.path.join(in_folder, pat)))
    files = sorted(set(files))

    if log_cb:
        log_cb(f"Found {len(files)} image(s).")

    for i, fp in enumerate(files, 1):
        src = Path(fp)
        out_path = Path(out_folder) / (src.stem + "_warped" + src.suffix)

        if out_path.exists() and not overwrite:
            if log_cb:
                log_cb(f"[{i}/{len(files)}] Skip exists: {out_path.name}")
            continue

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            if log_cb:
                log_cb(f"[{i}/{len(files)}] Failed to read: {src.name}")
            continue

        B = to_grayscale_float(img)
        Bp = preprocess_B(B, imgA, settingsB)
        warped = cv2.warpAffine(Bp, M, (W, H), flags=cv2.INTER_LINEAR)

        # Save 8-bit normalized for convenience
        w = warped.astype(np.float32)
        w -= w.min()
        mx = w.max()
        if mx > 0:
            w /= mx
        w8 = (w * 255).clip(0, 255).astype(np.uint8)

        ok = cv2.imwrite(str(out_path), w8)
        if log_cb:
            log_cb(f"[{i}/{len(files)}] {'Saved' if ok else 'FAILED'}: {out_path.name}")

# ----------------------------
# Batch apply video
# ----------------------------
def batch_apply_video(
    video_in: str,
    video_out: str,
    imgA: np.ndarray,
    settingsB: PreprocessBSettings,
    M: np.ndarray,
    log_cb=None,
):
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    H, W = imgA.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    # Choose codec based on extension
    ext = Path(video_out).suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        # mp4 is common; 'mp4v' is widely available
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(video_out, fourcc, fps, (W, H), isColor=False)
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Could not open output video writer.")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        B = to_grayscale_float(frame)
        Bp = preprocess_B(B, imgA, settingsB)
        warped = cv2.warpAffine(Bp, M, (W, H), flags=cv2.INTER_LINEAR)

        # VideoWriter expects 8-bit
        w = warped.astype(np.float32)
        w -= w.min()
        mx = w.max()
        if mx > 0:
            w /= mx
        w8 = (w * 255).clip(0, 255).astype(np.uint8)

        out.write(w8)

        if log_cb and (n % 25 == 0):
            log_cb(f"Processed {n} frame(s)...")

    cap.release()
    out.release()
    if log_cb:
        log_cb(f"Done. Total frames: {n}")


# ----------------------------
# Help dialog
# ----------------------------
class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help / How this app works")
        self.resize(700, 520)

        text = QtWidgets.QTextEdit(self)
        text.setReadOnly(True)
        text.setPlainText(
            "Concepts\n"
            "--------\n"
            "A = fixed/target image (defines the output canvas size).\n"
            "B = moving image (raw).\n"
            "B′ (B prime) = preprocessed B (resampled/padded/cropped depending on Mode).\n"
            "\n"
            "Preprocess Mode\n"
            "---------------\n"
            "none     : B′ = B (raw). Scale should typically be enabled.\n"
            "match_A  : B′ is resized into A's width/height.\n"
            "custom   : B′ is resized to a user-defined size.\n"
            "scale    : B′ is resized by a scale factor.\n"
            "\n"
            "Transform\n"
            "---------\n"
            "The computed matrix M always maps:\n"
            "    B′  ->  A\n"
            "So when you reuse M later, you must apply the SAME preprocessing to each new B frame to obtain B′,\n"
            "then warp B′ with M into A's canvas.\n"
            "\n"
            "Typical recommended workflow\n"
            "----------------------------\n"
            "1) Load A (defines output size)\n"
            "2) Load B\n"
            "3) Mode = match_A (often easiest)\n"
            "4) Pick corresponding points: click A then B′\n"
            "5) Compute transform\n"
            "6) Open QC view (RMSE + residual arrows + grid warp)\n"
            "7) Save session JSON (stores preprocess settings + matrix)\n"
            "8) Batch apply uses the same preprocess + matrix\n"
        )

        btn = QtWidgets.QPushButton("Close", self)
        btn.clicked.connect(self.accept)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(text)
        layout.addWidget(btn)

# ----------------------------
# Batch apply dialog
# ----------------------------
class BatchApplyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch apply (preprocess + transform)")
        self.resize(760, 320)

        self.in_folder = QtWidgets.QLineEdit()
        self.out_folder = QtWidgets.QLineEdit()
        self.pattern = QtWidgets.QLineEdit("*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")
        self.overwrite = QtWidgets.QCheckBox("Overwrite existing outputs")
        self.overwrite.setChecked(False)

        self.video_in = QtWidgets.QLineEdit()
        self.video_out = QtWidgets.QLineEdit()

        btn_in = QtWidgets.QPushButton("Browse…")
        btn_out = QtWidgets.QPushButton("Browse…")
        btn_vin = QtWidgets.QPushButton("Browse…")
        btn_vout = QtWidgets.QPushButton("Browse…")

        btn_in.clicked.connect(self._browse_in)
        btn_out.clicked.connect(self._browse_out)
        btn_vin.clicked.connect(self._browse_video_in)
        btn_vout.clicked.connect(self._browse_video_out)

        form = QtWidgets.QFormLayout()
        row_in = QtWidgets.QHBoxLayout(); row_in.addWidget(self.in_folder); row_in.addWidget(btn_in)
        row_out = QtWidgets.QHBoxLayout(); row_out.addWidget(self.out_folder); row_out.addWidget(btn_out)
        form.addRow("Input image folder", self._wrap(row_in))
        form.addRow("Output folder", self._wrap(row_out))
        form.addRow("Image patterns (; separated)", self.pattern)
        form.addRow("", self.overwrite)

        form.addRow(QtWidgets.QLabel("Video (optional)"))
        row_vin = QtWidgets.QHBoxLayout(); row_vin.addWidget(self.video_in); row_vin.addWidget(btn_vin)
        row_vout = QtWidgets.QHBoxLayout(); row_vout.addWidget(self.video_out); row_vout.addWidget(btn_vout)
        form.addRow("Input video file", self._wrap(row_vin))
        form.addRow("Output video file", self._wrap(row_vout))

        self.btn_run = QtWidgets.QPushButton("Run batch")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_cancel)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.log, 1)
        layout.addLayout(btns)

    def _wrap(self, hbox: QtWidgets.QHBoxLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(hbox)
        return w

    def _browse_in(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder")
        if p:
            self.in_folder.setText(p)

    def _browse_out(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if p:
            self.out_folder.setText(p)

    def _browse_video_in(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select input video", "", "Video (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        if p:
            self.video_in.setText(p)

    def _browse_video_out(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output video", "warped.mp4", "MP4 (*.mp4);;AVI (*.avi);;All files (*)")
        if p:
            self.video_out.setText(p)

    def append_log(self, msg: str):
        self.log.appendPlainText(msg)
        QtWidgets.QApplication.processEvents()


# ----------------------------
# UI Widgets
# ----------------------------
class ClickableImage(pg.GraphicsLayoutWidget):
    clicked_xy = QtCore.pyqtSignal(float, float)

    def __init__(self, title: str):
        super().__init__()
        self.view = self.addViewBox(lockAspect=True, invertY=True)
        self.view.setMouseEnabled(True, True)
        self.view.setMenuEnabled(False)

        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)

        self.overlay_item = pg.ImageItem()
        self.overlay_item.setOpacity(0.5)
        self.overlay_item.setVisible(False)
        self.view.addItem(self.overlay_item)

        self.scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(width=2))
        self.view.addItem(self.scatter)

        self.text_items: List[pg.TextItem] = []

        self.label = pg.LabelItem(justify="left")
        self.label.setText(title)
        self.addItem(self.label, row=0, col=0)

        # ✅ Robust clicking in PyQt5: use pyqtgraph's scene signal
        self.view.scene().sigMouseClicked.connect(self._on_scene_click)

    def _on_scene_click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        scene_pos = ev.scenePos()
        vb_pos = self.view.mapSceneToView(scene_pos)
        self.clicked_xy.emit(float(vb_pos.x()), float(vb_pos.y()))

    def set_image(self, img: Optional[np.ndarray]):
        if img is None:
            self.img_item.clear()
            return
        self.img_item.setImage(img, autoLevels=True)

    def set_overlay(self, img: Optional[np.ndarray], visible: bool):
        if img is None:
            self.overlay_item.clear()
            self.overlay_item.setVisible(False)
            return
        self.overlay_item.setImage(img, autoLevels=True)
        self.overlay_item.setVisible(bool(visible))

    def set_overlay_alpha(self, a: float):
        self.overlay_item.setOpacity(float(a))

    def set_points(self, pts: np.ndarray):
        for t in self.text_items:
            self.view.removeItem(t)
        self.text_items.clear()

        if pts is None or len(pts) == 0:
            self.scatter.setData([])
            return

        self.scatter.setData(pos=pts)
        for i, (x, y) in enumerate(pts):
            txt = pg.TextItem(text=str(i + 1), anchor=(0, 1))
            txt.setPos(float(x), float(y))
            self.view.addItem(txt)
            self.text_items.append(txt)


def make_grid_lines(width: int, height: int, step: int = 100) -> List[np.ndarray]:
    width = int(width)
    height = int(height)
    step = max(10, int(step))
    lines = []
    for x in range(0, width + 1, step):
        ys = np.linspace(0, height, num=50)
        xs = np.full_like(ys, x, dtype=float)
        lines.append(np.stack([xs, ys], axis=1))
    for y in range(0, height + 1, step):
        xs = np.linspace(0, width, num=50)
        ys = np.full_like(xs, y, dtype=float)
        lines.append(np.stack([xs, ys], axis=1))
    return lines


class QCWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QC / Sanity Check")
        self.resize(1200, 700)

        layout = QtWidgets.QGridLayout(self)

        self.plot_before_A = pg.PlotWidget(title="Before: A points")
        self.plot_before_B = pg.PlotWidget(title="Before: B′ points")
        self.plot_after = pg.PlotWidget(title="After: A vs T(B′) + residuals")
        self.plot_grid = pg.PlotWidget(title="Grid warp: B′ grid transformed into A space")

        for p in [self.plot_before_A, self.plot_before_B, self.plot_after, self.plot_grid]:
            p.showGrid(x=True, y=True, alpha=0.2)
            p.invertY(True)
            p.setAspectLocked(True)

        layout.addWidget(self.plot_before_A, 0, 0)
        layout.addWidget(self.plot_before_B, 0, 1)
        layout.addWidget(self.plot_after, 1, 0)
        layout.addWidget(self.plot_grid, 1, 1)

        self.info = QtWidgets.QLabel("")
        layout.addWidget(self.info, 2, 0, 1, 2)

    def update_qc(self, A_pts: np.ndarray, B_pts: np.ndarray, M: np.ndarray,
                  a_shape: Optional[Tuple[int, int]], b_shape: Optional[Tuple[int, int]],
                  grid_step: int = 120):
        for p in [self.plot_before_A, self.plot_before_B, self.plot_after, self.plot_grid]:
            p.clear()

        if A_pts is None or B_pts is None or len(A_pts) == 0 or len(A_pts) != len(B_pts):
            self.info.setText("No valid points to show.")
            return

        A_pts = np.asarray(A_pts, float)
        B_pts = np.asarray(B_pts, float)
        TB = apply_affine_2x3(M, B_pts)

        residual = A_pts - TB
        rmse = math.sqrt(np.mean(np.sum(residual**2, axis=1)))
        maxerr = float(np.max(np.sqrt(np.sum(residual**2, axis=1))))

        # Before plots
        self.plot_before_A.addItem(pg.ScatterPlotItem(pos=A_pts, size=10, pen=pg.mkPen(width=2)))
        self.plot_before_B.addItem(pg.ScatterPlotItem(pos=B_pts, size=10, pen=pg.mkPen(width=2)))

        for i, (x, y) in enumerate(A_pts):
            t = pg.TextItem(text=str(i + 1), anchor=(0, 1))
            t.setPos(float(x), float(y))
            self.plot_before_A.addItem(t)

        for i, (x, y) in enumerate(B_pts):
            t = pg.TextItem(text=str(i + 1), anchor=(0, 1))
            t.setPos(float(x), float(y))
            self.plot_before_B.addItem(t)

        # After overlay + residual vectors
        self.plot_after.addItem(pg.ScatterPlotItem(pos=A_pts, size=10, pen=pg.mkPen(width=2)))
        self.plot_after.addItem(pg.ScatterPlotItem(pos=TB, size=10, pen=pg.mkPen(width=2)))
        for i in range(len(A_pts)):
            x0, y0 = TB[i]
            x1, y1 = A_pts[i]
            self.plot_after.plot([x0, x1], [y0, y1], pen=pg.mkPen(width=1))

        # Grid warp
        if b_shape is not None:
            b_h, b_w = b_shape
            for poly in make_grid_lines(b_w, b_h, step=grid_step):
                poly_t = apply_affine_2x3(M, poly)
                self.plot_grid.plot(poly_t[:, 0], poly_t[:, 1], pen=pg.mkPen(width=1))

        if a_shape is not None:
            a_h, a_w = a_shape
            self.plot_after.setXRange(0, a_w)
            self.plot_after.setYRange(0, a_h)
            self.plot_grid.setXRange(0, a_w)
            self.plot_grid.setYRange(0, a_h)

        self.info.setText(f"RMSE: {rmse:.3f} px | Max error: {maxerr:.3f} px | N={len(A_pts)}")


# ----------------------------
# Main App
# ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procrustes Alignment (PyQt5)")
        self.resize(1500, 800)

        self.imgA: Optional[np.ndarray] = None
        self.imgB: Optional[np.ndarray] = None
        self.imgBprime: Optional[np.ndarray] = None

        self.settingsB = PreprocessBSettings()
        self.A_pts: List[List[float]] = []
        self.B_pts: List[List[float]] = []
        self.expecting = "A"

        self.M: Optional[np.ndarray] = None
        self.allow_scale = True

        self.viewA = ClickableImage("A (fixed / target)")
        self.viewB = ClickableImage("B (moving / preprocessed B′)")

        self.viewA.clicked_xy.connect(self.on_click_A)
        self.viewB.clicked_xy.connect(self.on_click_B)

        self.status = QtWidgets.QLabel("Load A and B to begin.")
        self.status.setWordWrap(True)

        # Buttons
        btn_loadA = QtWidgets.QPushButton("Load A…")
        btn_loadB = QtWidgets.QPushButton("Load B…")
        btn_loadA.clicked.connect(self.load_A)
        btn_loadB.clicked.connect(self.load_B)

        # Preprocess controls
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["none", "match_A", "custom", "scale"])
        self.cmb_mode.setCurrentText(self.settingsB.mode)

        self.spin_w = QtWidgets.QSpinBox()
        self.spin_h = QtWidgets.QSpinBox()
        self.spin_w.setRange(1, 20000)
        self.spin_h.setRange(1, 20000)
        self.spin_w.setValue(self.settingsB.custom_w)
        self.spin_h.setValue(self.settingsB.custom_h)

        self.dbl_scale = QtWidgets.QDoubleSpinBox()
        self.dbl_scale.setRange(0.01, 100.0)
        self.dbl_scale.setDecimals(3)
        self.dbl_scale.setSingleStep(0.1)
        self.dbl_scale.setValue(self.settingsB.scale)

        self.chk_keep_aspect = QtWidgets.QCheckBox("Keep aspect")
        self.chk_keep_aspect.setChecked(self.settingsB.keep_aspect)

        self.cmb_strategy = QtWidgets.QComboBox()
        self.cmb_strategy.addItems(["pad", "crop", "stretch"])
        self.cmb_strategy.setCurrentText(self.settingsB.aspect_strategy)

        self.cmb_interp = QtWidgets.QComboBox()
        self.cmb_interp.addItems(["area", "linear", "cubic", "lanczos", "nearest"])
        self.cmb_interp.setCurrentText(self.settingsB.interpolation)

        btn_apply_pre = QtWidgets.QPushButton("Apply preprocess")
        btn_apply_pre.clicked.connect(lambda: self.apply_preprocess(clear_points=True))

        # Points / transform
        btn_undo = QtWidgets.QPushButton("Undo last")
        btn_clear = QtWidgets.QPushButton("Clear points")
        btn_compute = QtWidgets.QPushButton("Compute transform")
        btn_qc = QtWidgets.QPushButton("Open QC view")

        btn_undo.clicked.connect(self.undo_last)
        btn_clear.clicked.connect(lambda: self.clear_points(silent=False))
        btn_compute.clicked.connect(self.compute_transform)
        btn_qc.clicked.connect(self.open_qc)

        btn_help = QtWidgets.QPushButton("Help…")
        btn_help.clicked.connect(self.open_help)

        btn_batch = QtWidgets.QPushButton("Batch apply…")
        btn_batch.clicked.connect(self.open_batch_apply)

        # Tooltips (nice + fixes the “?” confusion)
        btn_qc.setToolTip("Open QC plots: before/after points, residual vectors, and grid warp.")
        btn_help.setToolTip("How preprocessing + transform work, and how to reuse the matrix later.")
        btn_batch.setToolTip("Apply current preprocess + transform to a folder of images and/or a video.")


        self.chk_scale = QtWidgets.QCheckBox("Allow scale (similarity)")
        self.chk_scale.setChecked(True)
        self.chk_scale.stateChanged.connect(lambda _: self.set_allow_scale())

        # Preview
        self.chk_overlay = QtWidgets.QCheckBox("Show overlay (warp B′ → A)")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.stateChanged.connect(lambda _: self.update_preview())

        self.sld_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(50)
        self.sld_alpha.valueChanged.connect(lambda _: self.update_preview())

        # Save/load
        btn_save_json = QtWidgets.QPushButton("Save session JSON…")
        btn_load_json = QtWidgets.QPushButton("Load session JSON…")
        btn_save_warp = QtWidgets.QPushButton("Save warped image…")

        btn_save_json.clicked.connect(self.save_session_json)
        btn_load_json.clicked.connect(self.load_session_json)
        btn_save_warp.clicked.connect(self.save_warped_image)

        self.cmb_mode.currentTextChanged.connect(self.update_preprocess_state_ui)

        # Layout
        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(self.status)
        controls.addSpacing(8)
        controls.addWidget(btn_loadA)
        controls.addWidget(btn_loadB)
        controls.addWidget(btn_qc)
        controls.addWidget(btn_help)
        controls.addWidget(btn_batch)

        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Preprocess B"))
        controls.addWidget(QtWidgets.QLabel("Mode"))
        controls.addWidget(self.cmb_mode)

        wh_row = QtWidgets.QHBoxLayout()
        wh_row.addWidget(QtWidgets.QLabel("W"))
        wh_row.addWidget(self.spin_w)
        wh_row.addWidget(QtWidgets.QLabel("H"))
        wh_row.addWidget(self.spin_h)
        controls.addLayout(wh_row)

        sc_row = QtWidgets.QHBoxLayout()
        sc_row.addWidget(QtWidgets.QLabel("Scale"))
        sc_row.addWidget(self.dbl_scale)
        controls.addLayout(sc_row)

        controls.addWidget(self.chk_keep_aspect)
        controls.addWidget(QtWidgets.QLabel("Aspect strategy"))
        controls.addWidget(self.cmb_strategy)
        controls.addWidget(QtWidgets.QLabel("Interpolation"))
        controls.addWidget(self.cmb_interp)
        controls.addWidget(btn_apply_pre)

        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Points / Transform"))
        controls.addWidget(self.chk_scale)
        controls.addWidget(btn_compute)
        controls.addWidget(btn_qc)
        controls.addWidget(btn_undo)
        controls.addWidget(btn_clear)

        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Preview"))
        controls.addWidget(self.chk_overlay)
        controls.addWidget(QtWidgets.QLabel("Overlay alpha"))
        controls.addWidget(self.sld_alpha)

        controls.addSpacing(12)
        controls.addWidget(QtWidgets.QLabel("Export / Import"))
        controls.addWidget(btn_save_json)
        controls.addWidget(btn_load_json)
        controls.addWidget(btn_save_warp)

        controls.addStretch(1)

        control_widget = QtWidgets.QWidget()
        control_widget.setLayout(controls)

        center = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(center)
        lay.addWidget(self.viewA, 1)
        lay.addWidget(self.viewB, 1)
        lay.addWidget(control_widget, 0)
        self.setCentralWidget(center)

        self.qc_window: Optional[QCWindow] = None
        self.update_preprocess_state_ui()

    # ---- Loading
    def load_A(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load A (fixed)", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.status.setText("Failed to load A.")
            return
        self.imgA = to_grayscale_float(img)
        self.viewA.set_image(self.imgA)
        self.status.setText("Loaded A. Now load B (or re-apply preprocess if B is loaded).")
        if self.imgB is not None and self.cmb_mode.currentText() == "match_A":
            self.apply_preprocess(clear_points=True)

    def load_B(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load B (moving)", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.status.setText("Failed to load B.")
            return
        self.imgB = to_grayscale_float(img)
        self.apply_preprocess(clear_points=True)
        self.status.setText("Loaded B and applied preprocess. Click A then matching B′ point.")

    # ---- Preprocess
    def update_preprocess_state_ui(self):
        mode = self.cmb_mode.currentText()
        self.spin_w.setEnabled(mode == "custom")
        self.spin_h.setEnabled(mode == "custom")
        self.dbl_scale.setEnabled(mode == "scale")

    def apply_preprocess(self, clear_points: bool = True):
        if self.imgB is None:
            self.status.setText("Load B first.")
            return

        self.settingsB.mode = self.cmb_mode.currentText()
        self.settingsB.custom_w = int(self.spin_w.value())
        self.settingsB.custom_h = int(self.spin_h.value())
        self.settingsB.scale = float(self.dbl_scale.value())
        self.settingsB.keep_aspect = bool(self.chk_keep_aspect.isChecked())
        self.settingsB.aspect_strategy = self.cmb_strategy.currentText()
        self.settingsB.interpolation = self.cmb_interp.currentText()

        try:
            self.imgBprime = preprocess_B(self.imgB, self.imgA, self.settingsB)
        except Exception as e:
            self.status.setText(f"Preprocess failed: {e}")
            return

        self.viewB.set_image(self.imgBprime)

        if clear_points:
            self.clear_points(silent=True)
            self.status.setText("Preprocess applied. Points cleared. Click A then matching B′ point.")

        self.update_preview()

    # ---- Point picking
    def on_click_A(self, x: float, y: float):
        if self.imgA is None or self.imgBprime is None:
            self.status.setText("Load A and B first.")
            return
        if self.expecting != "A":
            self.status.setText("Now click the matching point in B′.")
            return
        self.A_pts.append([x, y])
        self.expecting = "B"
        self.refresh_points()
        self.status.setText(f"Picked A #{len(self.A_pts)}. Now click matching point in B′.")

    def on_click_B(self, x: float, y: float):
        if self.imgA is None or self.imgBprime is None:
            self.status.setText("Load A and B first.")
            return
        if self.expecting != "B":
            self.status.setText("Click a point in A first.")
            return
        self.B_pts.append([x, y])
        self.expecting = "A"
        self.refresh_points()
        self.status.setText(f"Picked B′ #{len(self.B_pts)}. Now click next point in A.")

    def refresh_points(self):
        A = np.array(self.A_pts, dtype=float) if self.A_pts else np.empty((0, 2))
        B = np.array(self.B_pts, dtype=float) if self.B_pts else np.empty((0, 2))
        self.viewA.set_points(A)
        self.viewB.set_points(B)

    def undo_last(self):
        if self.expecting == "B" and self.A_pts:
            self.A_pts.pop()
            self.expecting = "A"
        elif self.expecting == "A" and self.A_pts and self.B_pts:
            self.A_pts.pop()
            self.B_pts.pop()
            self.expecting = "A"
        self.refresh_points()
        self.M = None
        self.update_preview()
        self.status.setText("Undid last point(s).")

    def clear_points(self, silent: bool = False):
        self.A_pts.clear()
        self.B_pts.clear()
        self.expecting = "A"
        self.M = None
        self.refresh_points()
        self.update_preview()
        if not silent:
            self.status.setText("Cleared points. Click A then matching B′ point.")

    # ---- Transform + preview
    def set_allow_scale(self):
        self.allow_scale = bool(self.chk_scale.isChecked())

    def compute_transform(self):
        if self.imgA is None or self.imgBprime is None:
            self.status.setText("Load A and B first.")
            return
        if len(self.A_pts) < 2 or len(self.B_pts) < 2 or len(self.A_pts) != len(self.B_pts):
            self.status.setText("Need at least 2 matched pairs (and equal count in A and B′).")
            return

        A = np.array(self.A_pts, dtype=float)
        B = np.array(self.B_pts, dtype=float)

        try:
            R, s, t = similarity_transform(A, B, allow_scale=self.allow_scale)
            self.M = affine_2x3_from(R, s, t)
        except Exception as e:
            self.status.setText(f"Transform failed: {e}")
            self.M = None
            return

        self.status.setText(f"Transform computed from {len(A)} pairs. Toggle overlay / open QC.")
        self.update_preview()

    def update_preview(self):
        if self.imgA is None or self.imgBprime is None or self.M is None:
            self.viewA.set_overlay(None, visible=False)
            return

        show = bool(self.chk_overlay.isChecked())
        alpha = float(self.sld_alpha.value()) / 100.0

        H, W = self.imgA.shape[:2]
        warped = cv2.warpAffine(self.imgBprime, self.M, (W, H), flags=cv2.INTER_LINEAR)
        self.viewA.set_overlay(warped, visible=show)
        self.viewA.set_overlay_alpha(alpha)

    # ---- QC
    def open_qc(self):
        if self.M is None:
            self.status.setText("Compute transform first.")
            return
        if self.qc_window is None:
            self.qc_window = QCWindow(self)

        A = np.array(self.A_pts, dtype=float) if self.A_pts else np.empty((0, 2))
        B = np.array(self.B_pts, dtype=float) if self.B_pts else np.empty((0, 2))

        a_shape = self.imgA.shape[:2] if self.imgA is not None else None
        b_shape = self.imgBprime.shape[:2] if self.imgBprime is not None else None

        self.qc_window.update_qc(A, B, self.M, a_shape=a_shape, b_shape=b_shape, grid_step=120)
        self.qc_window.show()
        self.qc_window.raise_()
        self.qc_window.activateWindow()

    # ---- Save/load
    def session_dict(self) -> dict:
        return {
            "A_size": None if self.imgA is None else [int(self.imgA.shape[1]), int(self.imgA.shape[0])],
            "preprocess_B": asdict(self.settingsB),
            "points_A": self.A_pts,
            "points_Bprime": self.B_pts,
            "allow_scale": bool(self.allow_scale),
            "M_2x3": None if self.M is None else self.M.tolist(),
        }

    def save_session_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session JSON", "alignment_session.json", "JSON (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.session_dict(), f, indent=2)
        self.status.setText("Session saved.")

    def load_session_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.status.setText(f"Failed to load JSON: {e}")
            return

        pb = data.get("preprocess_B", {})
        for k, v in pb.items():
            if hasattr(self.settingsB, k):
                setattr(self.settingsB, k, v)

        self.cmb_mode.setCurrentText(self.settingsB.mode)
        self.spin_w.setValue(int(self.settingsB.custom_w))
        self.spin_h.setValue(int(self.settingsB.custom_h))
        self.dbl_scale.setValue(float(self.settingsB.scale))
        self.chk_keep_aspect.setChecked(bool(self.settingsB.keep_aspect))
        self.cmb_strategy.setCurrentText(self.settingsB.aspect_strategy)
        self.cmb_interp.setCurrentText(self.settingsB.interpolation)
        self.update_preprocess_state_ui()

        self.A_pts = data.get("points_A", []) or []
        self.B_pts = data.get("points_Bprime", []) or []
        self.allow_scale = bool(data.get("allow_scale", True))
        self.chk_scale.setChecked(self.allow_scale)

        M = data.get("M_2x3", None)
        self.M = np.array(M, dtype=float) if M is not None else None

        self.refresh_points()
        self.update_preview()
        self.status.setText("Session loaded. (Make sure the loaded A/B images match this session.)")

    def save_warped_image(self):
        if self.imgA is None or self.imgBprime is None or self.M is None:
            self.status.setText("Need A, B′, and a computed transform.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save warped image", "B_warped_to_A.png", "PNG (*.png);;TIFF (*.tif *.tiff);;All files (*)"
        )
        if not path:
            return

        H, W = self.imgA.shape[:2]
        warped = cv2.warpAffine(self.imgBprime, self.M, (W, H), flags=cv2.INTER_LINEAR)

        w = warped.copy()
        w = w - w.min()
        mx = w.max()
        if mx > 0:
            w = w / mx
        w8 = (w * 255).clip(0, 255).astype(np.uint8)

        ok = cv2.imwrite(path, w8)
        self.status.setText("Warped image saved." if ok else "Failed to save warped image.")

    def open_help(self):
        dlg = HelpDialog(self)
        dlg.exec_()


    def open_batch_apply(self):
        # Preconditions
        if self.imgA is None:
            self.status.setText("Batch apply requires A (output size). Load A first.")
            return
        if self.imgB is None:
            self.status.setText("Batch apply requires B loaded (preprocess settings context). Load B first.")
            return
        if self.M is None:
            self.status.setText("Batch apply requires a computed transform matrix. Compute transform first.")
            return

        dlg = BatchApplyDialog(self)

        def run():
            in_folder = dlg.in_folder.text().strip()
            out_folder = dlg.out_folder.text().strip()
            patterns = [p.strip() for p in dlg.pattern.text().split(";") if p.strip()]
            overwrite = bool(dlg.overwrite.isChecked())

            did_any = False

            # Images
            if in_folder and out_folder:
                did_any = True
                try:
                    dlg.append_log("=== Batch images ===")
                    batch_apply_images(
                        in_folder=in_folder,
                        out_folder=out_folder,
                        patterns=patterns,
                        imgA=self.imgA,
                        settingsB=self.settingsB,
                        M=self.M,
                        overwrite=overwrite,
                        log_cb=dlg.append_log,
                    )
                except Exception as e:
                    dlg.append_log(f"ERROR (images): {e}")

            # Video (optional)
            vin = dlg.video_in.text().strip()
            vout = dlg.video_out.text().strip()
            if vin and vout:
                did_any = True
                try:
                    dlg.append_log("=== Batch video ===")
                    batch_apply_video(
                        video_in=vin,
                        video_out=vout,
                        imgA=self.imgA,
                        settingsB=self.settingsB,
                        M=self.M,
                        log_cb=dlg.append_log,
                    )
                except Exception as e:
                    dlg.append_log(f"ERROR (video): {e}")

            if not did_any:
                dlg.append_log("Nothing to run: set (input folder + output folder) and/or (video in + video out).")

        dlg.btn_run.clicked.connect(run)
        dlg.exec_()



def main():
    pg.setConfigOptions(imageAxisOrder="row-major")
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
