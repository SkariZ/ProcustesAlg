# Procrustes-Based Image Alignment (PyQt5)

An interactive PyQt5 application for **point-based alignment of two image frames** using a **similarity (Procrustes) transform**, with built-in quality control and batch application to folders or videos.

This tool is designed for scientific imaging workflows where:
- one image (**A**) defines a fixed reference frame,
- another image (**B**) is transformed to match A,
- and the resulting transform must be **reusable and geometrically correct** for future frames.

---

## Key Concepts

### Images
- **A (fixed / target)**  
  Defines the *output coordinate system* and final image size.

- **B (moving / raw)**  
  The image to be aligned to A.

- **B′ (B prime)**  
  A *preprocessed* version of B (optional), used for point picking and alignment.

The computed transform **always maps**:
B′ → A


---

## Preprocessing Modes (B → B′)

Preprocessing controls **which coordinate system** the alignment is computed in.

| Mode        | Description |
|-------------|------------|
| `none`      | B′ = raw B (no resampling). The transform includes scale. |
| `match_A`   | B is resampled to A’s width/height before alignment. |
| `custom`    | B is resampled to a user-defined size. |
| `scale`     | B is resampled by a scale factor. |

Additional options:
- keep aspect ratio
- pad / crop / stretch
- interpolation method

### Important Rule
> **When reusing a transform later, the exact same preprocessing must be applied.**

This guarantees correctness and reproducibility.

---

## Transform Model

The app computes a **similarity transform**:

- translation  
- rotation  
- optional uniform scale  

using corresponding point pairs selected interactively.

Internally, the transform is stored as a 2×3 affine matrix compatible with `cv2.warpAffine`.

---

## Typical Workflow

1. Load **A** (fixed image)
2. Load **B** (moving image)
3. Choose preprocessing mode for B
   - `match_A` is recommended for easiest point picking
4. Pick corresponding points:
   - click a point in A
   - click the matching point in B′
5. Compute transform
6. Inspect results using **QC view**
7. Save session (JSON)
8. Apply the same preprocessing + transform to:
   - a folder of images
   - a video

---

## Quality Control (QC View)

The QC window provides:

- **Before**: A points and B′ points (indexed)
- **After**: A points vs. transformed B′ points
- **Residual vectors** for each correspondence
- **RMSE and max error (pixels)**
- **Grid warp visualization** (intuitive sanity check)

This makes it easy to detect:
- incorrect correspondences
- flipped axes
- scale or rotation mistakes

---

## Batch Application

The batch tool applies the **current preprocessing settings and transform** to:

- all images in a folder
- a video (frame by frame)

Pipeline:
B_raw → preprocess → B′ → warpAffine(M) → A-sized output

Batch output is guaranteed to match the interactive alignment, as long as the same preprocessing is used.

---

## Saved Session Format

A saved session (`.json`) contains:

- A output size
- preprocessing settings for B
- picked point coordinates
- similarity transform matrix (2×3)
- scale enabled/disabled flag

This makes alignment **fully reproducible** and suitable for automated processing.

---

## Installation

```bash
pip install PyQt5 pyqtgraph opencv-python numpy
