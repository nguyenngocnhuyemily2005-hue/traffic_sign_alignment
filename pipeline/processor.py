# pipeline/processor.py
# Main processor for the Traffic Sign Alignment Pipeline

import os
import cv2
from config import ENABLED_STEPS, INPUT_DIR, OUTPUT_DIR
from utils.io import load_image, save_image, get_image_list, load_label
from steps.crop import auto_crop
from steps.resize import resize_image
from steps.enhance import enhance_contrast
from steps.sharpen import sharpen_image
from steps.align import align_image

# all steps in order
class TrafficSignProcessor:
    def __init__(self):
        self.steps = {
            'crop': self._crop_step,
            'resize': self._resize_step,
            'enhance': self._enhance_step,
            'sharpen': self._sharpen_step,
            'align': self._align_step
        }

    # currently use auto_crop as default, but can be replaced with manual cropping if needed
    def _crop_step(self, image):
        return auto_crop(image)

    def _resize_step(self, image):
        return resize_image(image)

    def _enhance_step(self, image):
        return enhance_contrast(image)

    def _sharpen_step(self, image):
        return sharpen_image(image)

    def _align_step(self, image):
        # For alignment, we might need a reference image
        # For now, assume no reference or use the first image as reference
        return align_image(image)

    def process_image(self, image_path, label_path=None):
        """Process a single image through all enabled steps."""
        image = load_image(image_path)
        labels = load_label(label_path)

        for step_name in ENABLED_STEPS:
            if step_name in self.steps:
                image = self.steps[step_name](image)

        filename = os.path.basename(image_path)
        output_path = save_image(image, f"processed_{filename}")
        return output_path

    def run_pipeline(self):
        """Run the pipeline on all images in the input directory."""
        pairs = get_image_list()
        if not pairs:
            print("No images found in input directory.")
            return

        print(f"Processing {len(pairs)} images...")
        for image_path, label_path in pairs:
            try:
                output_path = self.process_image(image_path, label_path)
                print(f"Processed: {image_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        print("Pipeline completed.")