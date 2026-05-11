# Traffic Sign Alignment Pipeline

A modular Computer Vision pipeline for traffic sign detection and perspective alignment using classical image processing techniques.

This project focuses on:
- Region of Interest (ROI) extraction
- HSV color filtering
- Morphological processing
- Edge and contour detection
- Shape analysis
- Perspective rectification (homography)

The pipeline is designed for educational and experimental purposes in Computer Vision.

---

# Features

- Modular step-by-step pipeline
- Classical Computer Vision approach (no deep learning)
- Easy debugging and visualization
- Standalone testing scripts for each step
- Batch image processing support
- Perspective correction for traffic signs

---

# Current Pipeline

```text
Input Image
    ↓
STEP 1 — ROI Extraction
    ↓
STEP 2 — HSV Color Filtering
    ↓
STEP 3 — Morphology & Denoising
    ↓
STEP 4 — Edge & Contour Detection
    ↓
STEP 5 — Shape Analysis
    ↓
STEP 6 — Perspective Rectification
    ↓
STEP 7 — Image Enhancement
```

---

# Project Structure

```text
traffic_sign_alignment/
│
├── config.py
├── main.py
├── README.md
│
├── input_images/
│   └── sign1.jpg
│
├── output_images/
│
├── standalone_demo/
│   ├── test_step01_roi.py
│   ├── test_step02_hsv_filter.py
│   ├── test_step03_morphology.py
│   └── test_step04_edge_contour.py
│
├── steps/
│   ├── step01_roi.py
│   ├── step02_hsv_filter.py
│   ├── step03_morphology.py
│   ├── step04_edge_contour.py
│   ├── step05_shape_analysis.py
│   ├── step06_rectify.py
│   └── step07_enhance.py
│
├── pipeline/
│   └── processor.py
│
└── utils/
    └── io.py
```

---

# Installation

## 1. Clone repository

```bash
git clone https://github.com/yourusername/traffic_sign_alignment.git

cd traffic_sign_alignment
```

---

## 2. Install dependencies

```bash
pip install opencv-python numpy matplotlib
```

---

# Usage

## Run standalone demos

Used for debugging and visualizing each individual step.

Example:

```bash
python standalone_demo/test_step01_roi.py
```

or

```bash
python standalone_demo/test_step02_hsv_filter.py
```

---

## Run full pipeline

```bash
python main.py
```

Processed outputs will be saved inside:

```text
output_images/
```

---

# Pipeline Steps

## STEP 1 — ROI Extraction

Goal:
- Reduce unnecessary background
- Focus on likely traffic sign regions

Current method:
- Right-side crop
- Upper-region crop
- Simple heuristic ROI

---

## STEP 2 — HSV Color Filtering

Goal:
- Extract traffic sign colors

Current targets:
- Blue traffic signs
- Red traffic signs

Techniques:
- RGB → HSV conversion
- Color thresholding
- Binary mask generation

---

## STEP 3 — Morphology & Denoising

Goal:
- Remove noise from masks
- Connect fragmented regions

Techniques:
- Morphological Closing

---

## STEP 4 — Edge & Contour Detection

Goal:
- Detect candidate traffic sign regions

Techniques:
- Canny Edge Detection
- Contour Extraction
- Area filtering

---

## STEP 5 — Shape Analysis

Goal:
- Determine whether a contour resembles a traffic sign

Techniques:
- Polygon approximation
- Shape classification
- Rectangle / Triangle / Circle detection
- Geometric filtering

---

## STEP 6 — Perspective Rectification

Goal:
- Align traffic signs into a front-facing view

Techniques:
- Corner extraction
- Homography
- Perspective transform
- warpPerspective()

---

## STEP 7 — Enhancement

Goal:
- Improve visual quality of aligned signs

Possible techniques:
- Sharpening
- Contrast enhancement
- Histogram equalization

---

# Configuration

Main configuration file:

```text
config.py
```

Example:

```python
ENABLED_STEPS = [
    'roi',
    'hsv_filter',
    'morphology',
    'edge_contour',
    'shape_analysis',
    'rectify',
    'enhance'
]
```

---

# Requirements

- Python 3.11+
- OpenCV
- NumPy
- Matplotlib

---

# Notes

This project intentionally uses traditional Computer Vision techniques instead of deep learning models in order to:
- better understand image processing fundamentals
- visualize each processing stage
- study geometric transformations and contour analysis

---

# Future Improvements

Possible future upgrades:
- Better ROI estimation
- Night-condition robustness
- Circular sign handling
- Adaptive HSV thresholds
- Lane detection integration
- Machine learning classification

---

# License

MIT License