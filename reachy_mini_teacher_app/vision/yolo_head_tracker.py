"""
    # YOLO Head Tracker: Vision-Based Face Tracking

    The `HeadTracker` class provides a lightweight, vision-based system for the Reachy Mini to "see" and locate a human user. It uses a **YOLOv11** (You Only Look Once) face detection model to identify the most prominent face in a camera feed and translate its position into coordinates the robot can use for "gaze" or "follow" behaviors.

    ---

    ## 👁️ Core Concept: Vision for Interaction

    While the `HeadWobbler` handles how the robot moves while *speaking*, the `HeadTracker` handles how the robot *reacts* to the user's presence. It identifies a face and calculates its center point relative to the camera's field of view.

    ### Why YOLO?

    * **Speed:** YOLO is designed for real-time inference, meaning the robot can track your head with minimal lag.
    * **Robustness:** This specific model (`YOLOv11n-face-detection`) is optimized to find faces even in varied lighting or at slight angles.
    * **Distance Filtering:** The code includes logic to prefer "larger" faces, ensuring the robot talks to the person standing directly in front of it rather than someone in the background.

    ---

    ## 🛠️ Features

    * **Automatic Model Management:** Downloads the optimized weights directly from the HuggingFace Hub.
    * **Best Face Selection:** If multiple people are in the frame, it uses a weighted score (70% Confidence / 30% Size) to lock onto the primary user.
    * **Normalized Coordinates:** Converts raw pixel data (e.g., "Pixel 320") into a standardized range of **[-1, 1]**.
    * `-1.0`: Far Left / Top
    * `0.0`: Dead Center
    * `1.0`: Far Right / Bottom


    * **Device Flexibility:** Automatically runs on **CPU** or **CUDA (GPU)** if available.

    ---

    ## 📐 Coordinate Mapping

    The tracker transforms the camera image into a "MediaPipe-style" coordinate system. This makes it incredibly easy to map "Where the human is" to "Where the robot should look."

    range]

    * **Center X:** `(x_min + x_max) / 2` mapped to `[-1, 1]`
    * **Center Y:** `(y_min + y_max) / 2` mapped to `[-1, 1]`

    ---

    ## 🚀 Usage

    ```python
    import cv2
    from head_tracker import HeadTracker

    # 1. Initialize the tracker
    tracker = HeadTracker(device="cpu", confidence_threshold=0.5)

    # 2. Get a frame from your camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # 3. Get the head position
    # Returns: (NDArray[x, y], roll_angle)
    position, roll = tracker.get_head_position(frame)

    if position is not None:
        print(f"User is at X: {position[0]:.2f}, Y: {position[1]:.2f}")
        # You can now send these values to Reachy's neck controllers!

    ```

    ---

    ## 🧠 Logical Flow

    1. **Inference:** The image is fed into the YOLO model.
    2. **Detection Filtering:** Any detections below the `confidence_threshold` are discarded.
    3. **Scoring:** Valid faces are ranked. A face that is slightly less confident but much **larger** (closer to the camera) will be chosen over a small, distant face with high confidence.
    4. **Normalization:** The bounding box center is calculated and scaled relative to the image width and height.

    ---

    ## 🔧 Component Breakdown

    | Method | Responsibility |
    | --- | --- |
    | `__init__` | Downloads the `.pt` model file and loads it into memory. |
    | `_select_best_face` | The "social" filter. It ensures the robot doesn't get distracted by background faces. |
    | `_bbox_to_mp_coords` | The "translator." Converts pixel values into the `[-1, 1]` math range. |
    | `get_head_position` | The public API. Coordinates the entire process from image to location. |

    ---

    ### ⚠️ Dependencies

    This module requires the `yolo_vision` extra. You can install it via:

    ```bash
    pip install '.[yolo_vision]'

    ```

"""
from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


try:
    from supervision import Detections
    from ultralytics import YOLO  # type: ignore
except ImportError as e:
    raise ImportError(
        "To use YOLO head tracker, please install the extra dependencies: pip install '.[yolo_vision]'",
    ) from e
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)


class HeadTracker:
    """Lightweight head tracker using YOLO for face detection."""

    def __init__(
        self,
        model_repo: str = "AdamCodd/YOLOv11n-face-detection",
        model_filename: str = "model.pt",
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        """Initialize YOLO-based head tracker.

        Args:
            model_repo: HuggingFace model repository
            model_filename: Model file name
            confidence_threshold: Minimum confidence for face detection
            device: Device to run inference on ('cpu' or 'cuda')

        """
        self.confidence_threshold = confidence_threshold

        try:
            # Download and load YOLO model
            model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
            self.model = YOLO(model_path).to(device)
            logger.info(f"YOLO face detection model loaded from {model_repo}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _select_best_face(self, detections: Detections) -> int | None:
        """
        Selects the primary face to track using a weighted scoring system of confidence and size.

        WHAT IT DOES:
        1. Filters out "ghost" detections that are below the confidence threshold.
        2. Calculates the physical area (size) of every remaining face.
        3. Assigns a score to each face based on both how sure the AI is (70%) 
           and how close/large the face appears (30%).
        4. Returns the index of the highest-scoring face.

        WHY IT IS NEEDED:
        A robot shouldn't just track the "most confident" face. A person far away might 
        have 99% confidence but be irrelevant to the conversation. Conversely, a person 
        right in front of the robot might have 85% confidence due to lighting. By 
        factoring in 'Area', we ensure the robot prioritizes the person closest 
        to its sensors (the 'Largest' face).
        """
        # 1. EARLY EXIT: If the AI found zero boxes, stop immediately.
        if detections.xyxy.shape[0] == 0 or detections.confidence is None:
            return None

        # 2. CONFIDENCE FILTER: Create a 'mask' (True/False list) to ignore low-quality guesses.
        # This removes background noise or objects that barely look like faces.
        valid_mask = detections.confidence >= self.confidence_threshold
        if not np.any(valid_mask):
            return None

        # 3. INDEX MAPPING: Get the actual positions of the 'Good' detections.
        valid_indices = np.where(valid_mask)[0]
        
        # 4. AREA CALCULATION: Measure size (Width * Height) for each valid face.
        # boxes[:, 2] - boxes[:, 0] is (x_max - x_min)
        # boxes[:, 3] - boxes[:, 1] is (y_max - y_min)
        boxes = detections.xyxy[valid_indices]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 5. DATA EXTRACTION: Get the confidence scores for only the valid faces.
        confidences = detections.confidence[valid_indices]
        
        # 6. WEIGHTED SCORING:
        # We normalize the area (areas / np.max(areas)) so it's a value between 0 and 1.
        # 0.7 Weight: Prioritize accuracy/certainty.
        # 0.3 Weight: Prioritize physical proximity (larger = closer).
        scores = (confidences * 0.7) + ((areas / np.max(areas)) * 0.3)
        
        # 7. FINAL SELECTION: Find the highest score and map it back to the original detection list.
        best_idx = valid_indices[np.argmax(scores)]
        return int(best_idx)

    def _bbox_to_mp_coords(self, bbox: NDArray[np.float32], w: int, h: int) -> NDArray[np.float32]:
        """Convert bounding box center to MediaPipe-style coordinates [-1, 1].

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            w: Image width
            h: Image height

        Returns:
            Center point in [-1, 1] coordinates

        """
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        # Normalize to [0, 1] then to [-1, 1]
        norm_x = (center_x / w) * 2.0 - 1.0
        norm_y = (center_y / h) * 2.0 - 1.0

        return np.array([norm_x, norm_y], dtype=np.float32)

    def get_head_position(self, img: NDArray[np.uint8]) -> Tuple[NDArray[np.float32] | None, float | None]:
        """Get head position from face detection.

        Args:
            img: Input image

        Returns:
            Tuple of (eye_center [-1,1], roll_angle)

        """
        h, w = img.shape[:2]

        try:
            # Run YOLO inference
            results = self.model(img, verbose=False)
            detections = Detections.from_ultralytics(results[0])

            # Select best face
            face_idx = self._select_best_face(detections)
            if face_idx is None:
                logger.debug("No face detected above confidence threshold")
                return None, None

            bbox = detections.xyxy[face_idx]

            if detections.confidence is not None:
                confidence = detections.confidence[face_idx]
                logger.debug(f"Face detected with confidence: {confidence:.2f}")

            # Get face center in [-1, 1] coordinates
            face_center = self._bbox_to_mp_coords(bbox, w, h)

            # Roll is 0 since we don't have keypoints for precise angle estimation
            roll = 0.0

            return face_center, roll

        except Exception as e:
            logger.error(f"Error in head position detection: {e}")
            return None, None

    def get_head_position_with_bbox(self, img: NDArray[np.uint8]):
        """Get head position from face detection.

        Args:
            img: Input image

        Returns:
            Tuple of (eye_center [-1,1], roll_angle, bbox)

        """
        h, w = img.shape[:2]
        try:
            results = self.model(img, verbose=False)
            detections = Detections.from_ultralytics(results[0])
            face_idx = self._select_best_face(detections)
            
            if face_idx is None:
                return None, None, None

            bbox = detections.xyxy[face_idx] # [x1, y1, x2, y2]
            face_center = self._bbox_to_mp_coords(bbox, w, h)
            return face_center, 0.0, bbox
        except Exception as e:
            return None, None, None