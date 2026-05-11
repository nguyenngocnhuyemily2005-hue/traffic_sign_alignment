# Traffic Sign Alignment Pipeline

A traditional Computer Vision pipeline for detecting and analyzing Vietnamese traffic signs using image processing techniques.

This project focuses on building a modular traffic sign detection pipeline step-by-step using:

* ROI extraction
* HSV color filtering
* Morphological operations
* Contour analysis
* Polygon approximation
* Semantic shape filtering

The pipeline is designed mainly for experimentation, debugging, and understanding classical Computer Vision methods before moving into more advanced approaches.

---

# Features

* Modular processing pipeline
* Traditional Computer Vision workflow
* HSV-based traffic sign color detection
* Morphological noise reduction
* Polygon approximation using contours
* Semantic shape classification:

  * Triangle
  * Rectangle
  * Circle
* Visualization for every processing step
* Separate testing environment using standalone demo scripts

---

# Project Structure

```text
traffic_sign_alignment/
│
├── input_images/
│   └── sign images for testing
│
├── output_images/
│   └── generated outputs from each step
│
├── standalone_demo/
│   ├── test_step01_roi.py
│   ├── test_step02_hsv_filter.py
│   ├── test_step03_morphology.py
│   └── test_step04_contour.py
│
├── steps/
│   ├── step01_roi.py
│   ├── step02_hsv_filter.py
│   ├── step03_morphology.py
│   └── step04_contour.py
│
├── README.md
├── requirements.txt
└── main.py
```

---

# Pipeline Overview

## STEP 1 — ROI Extraction

Simple heuristic ROI extraction using:

* Right-side crop
* Upper-region crop

Goal:
Reduce unnecessary background and focus on roadside traffic signs.

---

## STEP 2 — HSV Color Filtering

Traffic sign color segmentation using HSV color space.

Detected colors:

* Blue traffic signs
* Red traffic signs

Techniques used:

* Gaussian blur
* HSV conversion
* Color thresholding
* Binary mask generation

---

## STEP 3 — Morphology + Area Filtering

Mask cleanup and noise reduction.

Techniques used:

* Morphological closing
* Connected component analysis
* Area filtering

Goal:
Preserve meaningful traffic sign regions while removing small noise blobs.

---

## STEP 4 — Semantic Contour Filtering

Traffic-sign-like shape detection using contour analysis.

Techniques used:

* Contour extraction
* Polygon approximation
* Circularity analysis
* Aspect ratio filtering
* Fill ratio filtering
* Shape classification

Supported shape types:

* Triangle
* Rectangle
* Circle

---

# Installation

## Clone repository

```bash
git clone https://github.com/yourusername/traffic-sign-alignment.git

cd traffic-sign-alignment
```

---

## Create virtual environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install opencv-python matplotlib numpy
```

---

# Usage

Place traffic sign images inside:

```text
input_images/
```

Run a testing script:

```bash
python standalone_demo/test_step04_contour.py
```

Outputs will be saved to:

```text
output_images/
```

---

# Requirements

* Python 3.9+
* OpenCV
* NumPy
* Matplotlib

---

# Current Status

Implemented:

* ROI extraction
* HSV filtering
* Morphology cleanup
* Semantic contour detection

Planned:

* Perspective rectification
* Traffic sign alignment
* OCR / text extraction
* Classification improvements
* Deep learning integration

---

# Notes

This project intentionally focuses on classical Computer Vision techniques instead of deep learning in the early stages.

The goal is to better understand:

* image preprocessing
* contour behavior
* morphology operations
* geometric filtering
* shape analysis

before integrating more advanced models.

---

# License

This project is for educational and research purposes.
