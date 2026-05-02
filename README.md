# Traffic Sign Alignment Pipeline

An image processing pipeline designed specifically for traffic sign alignment. This project provides a modular framework for preprocessing traffic sign images through multiple enhancement and alignment steps.

## Features

- **Modular Pipeline**: Configurable processing steps that can be enabled/disabled
- **Image Processing Steps**:
  - Cropping: Automatic border removal
  - Resizing: Standardize image dimensions
  - Enhancement: Contrast and brightness adjustment
  - Sharpening: Improve image clarity
  - Alignment: Feature-based image alignment
- **Batch Processing**: Process multiple images automatically
- **Extensible Architecture**: Easy to add new processing steps

## Project Structure

```
traffic_sign_pipeline/
│
├── config.py                 # Configuration settings and parameters
├── main.py                   # Main entry point for running the pipeline
│
├── utils/
│   └── io.py                 # Input/output utilities for image handling
│
├── steps/
│   ├── crop.py               # Image cropping functionality
│   ├── resize.py             # Image resizing operations
│   ├── enhance.py            # Image enhancement (contrast, brightness)
│   ├── sharpen.py            # Image sharpening algorithms
│   └── align.py              # Image alignment using feature matching
│
└── pipeline/
    └── processor.py          # Main pipeline orchestrator
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-alignment.git
   cd traffic-sign-alignment
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```

3. Set up input directory:
   ```bash
   mkdir -p traffic_sign_pipeline/input_images
   # Place your traffic sign images in this directory
   ```

## Usage

### Basic Usage

1. Place your traffic sign images in the `/input_images/` directory.

2. Run the pipeline:
   ```bash
   cd traffic_sign_alignment
   python main.py
   ```

3. Processed images will be saved in `traffic_sign_pipeline/output_images/`.

### Configuration

Modify `config.py` to customize:
- Input/output directories
- Image processing parameters
- Enabled processing steps

Example configuration:
```python
# config.py
DEFAULT_IMAGE_SIZE = (224, 224)
SHARPEN_FACTOR = 1.5
ENABLED_STEPS = ['crop', 'resize', 'enhance', 'sharpen', 'align']
```

### Advanced Usage

You can also use individual processing steps programmatically:

```python
from steps.crop import auto_crop
from steps.align import align_image
from utils.io import load_image, save_image

# Load and process an image
image = load_image('path/to/image.jpg')
cropped = auto_crop(image)
aligned = align_image(cropped, reference_image)
save_image(aligned, 'aligned_image.jpg')
```

## Processing Steps

### 1. Crop
Automatically removes borders and focuses on the main content of traffic signs.

### 2. Resize
Standardizes image dimensions for consistent processing.

### 3. Enhance
Improves image quality through contrast adjustment and brightness correction.

### 4. Sharpen
Applies sharpening filters to improve edge definition.

### 5. Align
Uses feature matching algorithms to align images, correcting for rotation and perspective.

## Requirements

- Python 3.7+
- OpenCV 4.0+
- NumPy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your processing step or improvement
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
