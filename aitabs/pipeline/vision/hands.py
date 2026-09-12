"""MediaPipe hand tracking — 21 finger landmarks per hand, per frame."""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class Landmark(TypedDict):
    id: int        # MediaPipe landmark index 0–20
    x: float       # normalized [0, 1] relative to frame width
    y: float       # normalized [0, 1] relative to frame height
    z: float       # depth relative to wrist (negative = closer to camera)


class HandResult(TypedDict):
    handedness: str          # "Left" or "Right"
    landmarks: list[Landmark]
    score: float             # detection confidence


def build_detector(
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.5,
):
    """Create and return a MediaPipe Hands detector.

    Call this once and reuse across frames — MediaPipe initializes a TF-Lite
    model internally and re-creating it per frame is expensive.

    Args:
        max_num_hands: Set to 1 if you only need the fretting hand (left for
            right-handed players). 2 is safer for detecting both hands.
        min_detection_confidence: Threshold for the palm detection model.
        min_tracking_confidence: Threshold for the landmark tracking model.
            Lower = keeps tracking through more partial occlusions.

    Returns:
        mediapipe.solutions.hands.Hands context manager.

    TODO (you):
        For your recording setup, experiment with whether you need both hands
        or just the fretting hand. Detecting both adds ~30% latency.
    """
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return hands


def track_hands(frame: np.ndarray, detector=None) -> list[HandResult]:
    """Run MediaPipe Hands on a single video frame.

    Args:
        frame: BGR uint8 numpy array from cv2.VideoCapture.
        detector: MediaPipe Hands instance from build_detector().
                  If None, a new one is created (slow — don't do this per frame).

    Returns:
        List of HandResult dicts. Empty if no hands detected.

    TODO (you):
        - Print the landmark positions on a few frames to verify the y-axis
          direction. MediaPipe uses top-left origin, same as OpenCV.
        - The fretting hand tip landmarks you care most about are:
            4  = thumb tip
            8  = index finger tip
            12 = middle finger tip
            16 = ring finger tip
            20 = pinky tip
          These are the fret-pressing fingertips. The base knuckles (5, 9, 13, 17)
          tell you which fret position the hand is anchored in.
    """
    raise NotImplementedError(
        "Implement track_hands:\n"
        "1. Convert BGR frame to RGB (MediaPipe requires RGB)\n"
        "2. Call detector.process(rgb_frame)\n"
        "3. If results.multi_hand_landmarks is None, return []\n"
        "4. Zip results.multi_hand_landmarks with results.multi_handedness\n"
        "5. Build HandResult dicts and return"
    )


def fingertip_to_string_fret(
    landmark: Landmark,
    fretboard_coords: dict,
) -> tuple[int, int] | None:
    """Map a fingertip landmark to a (string, fret) position.

    Args:
        landmark: A single fingertip Landmark (e.g. index tip, id=8).
        fretboard_coords: Output of fretboard.fretboard_coordinates().

    Returns:
        (string_index, fret) tuple, or None if the fingertip is outside
        the fretboard bounding box.

    TODO (you):
        This is the coordinate transform that connects vision to tab output.
        The key insight: if you know the pixel (x, y) of a fingertip and you
        know the pixel positions of each fret and string from fretboard_coords,
        you can find the nearest fret and string by Euclidean distance or
        simple bin assignment. Start with bin assignment (divide box into grid),
        then refine to nearest-neighbor if you see accuracy problems.
    """
    raise NotImplementedError(
        "Implement fingertip_to_string_fret — see docstring for the approach"
    )
