"""RT-DETR fretboard detection — locate the guitar neck in a video frame."""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class FretboardBox(TypedDict):
    x1: float   # left edge in pixels
    y1: float   # top edge in pixels
    x2: float   # right edge in pixels
    y2: float   # bottom edge in pixels
    score: float


def load_model(checkpoint: str = "PekingU/rtdetr_r50vd"):
    """Load an RT-DETR model from a HuggingFace checkpoint.

    For the fine-tuned version, replace checkpoint with the path to your
    Roboflow-trained model exported in HuggingFace format.

    Returns:
        (model, processor) tuple ready for inference.

    TODO (you):
        1. Start with the pre-trained PekingU/rtdetr_r50vd checkpoint to verify
           the integration works end-to-end.
        2. Export your Roboflow dataset in COCO format, then fine-tune using
           the HuggingFace Trainer or the Roboflow-provided training script.
        3. After fine-tuning, save your checkpoint locally and update this path.
    """
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    model = RTDetrForObjectDetection.from_pretrained(checkpoint)
    model.eval()
    return model, processor


def detect_fretboard(
    frame: np.ndarray,
    model=None,
    processor=None,
    confidence_threshold: float = 0.7,
) -> FretboardBox | None:
    """Detect the guitar fretboard bounding box in a single video frame.

    Args:
        frame: BGR uint8 numpy array from cv2.VideoCapture.
        model: RTDetrForObjectDetection instance (load once, reuse per frame).
        processor: RTDetrImageProcessor instance.
        confidence_threshold: Detections below this score are ignored.

    Returns:
        FretboardBox with pixel coordinates, or None if no fretboard found.

    TODO (you):
        1. Run this on a few sample frames from your recording sessions to
           verify the bounding box is tight around the fretboard.
        2. The fretboard class ID in your fine-tuned model will be 0 (or whatever
           Roboflow assigns). Update `target_label` accordingly.
        3. For a fixed-camera setup, you can cache the bounding box from the
           first frame and only re-run RT-DETR every N frames to save compute.
    """
    raise NotImplementedError(
        "Implement detect_fretboard:\n"
        "1. Convert BGR frame to PIL Image\n"
        "2. Run processor + model\n"
        "3. Filter results by label (fretboard class) and score threshold\n"
        "4. Return the highest-confidence box as a FretboardBox"
    )


def fretboard_coordinates(
    box: FretboardBox,
    num_strings: int = 6,
    num_frets: int = 22,
) -> dict:
    """Compute pixel positions of each string and fret intersection.

    Given the bounding box of the fretboard, estimate where each string
    and fret line appears in the image. This is used to map finger landmarks
    (from MediaPipe) to string/fret positions.

    Args:
        box: Fretboard bounding box from detect_fretboard.
        num_strings: Number of strings visible (typically 6).
        num_frets: Number of frets to divide the box into.

    Returns:
        Dict with:
          'strings': list of y-coordinates (one per string, top to bottom)
          'frets':   list of x-coordinates (one per fret, nut to body)

    TODO (you):
        This function assumes the fretboard is roughly rectangular and
        axis-aligned. For angled or perspective-distorted frames, you'll
        need a homography transform instead. OpenCV's getPerspectiveTransform
        is your friend here — use the four corners of the bounding box.
    """
    raise NotImplementedError(
        "Implement fretboard_coordinates — divide the bounding box into a grid"
    )
