#!/usr/bin/env python3
"""
Rocky Gesture Recognition System v2.0

A robust real-time computer vision system for detecting and classifying
hand gestures and facial expressions using MediaPipe and OpenCV.

Features:
- Eye Aspect Ratio (EAR) for wink detection
- HSV color space analysis for tongue detection
- Haar Cascade for sunglasses detection
- Hand landmark geometry for finger-gun gesture
- Temporal smoothing via majority vote algorithm
- Configurable detection thresholds
- Error handling and graceful degradation

Author: Advanced CV Team
Version: 2.0.0
Python: 3.11+
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Deque
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

CONFIG = {
    'eye': {
        'ear_thresh': 0.18,      # EAR below this = eye closed
        'diff_thresh': 0.06      # Difference between eyes to detect wink
    },
    'sunglasses': {
        'brightness_thresh': 85  # Mean brightness threshold for dark glasses
    },
    'tongue': {
        'mouth_open_thresh': 0.35,  # Mouth open ratio
        'red_ratio_thresh': 0.18     # % of red/pink pixels in mouth ROI
    },
    'gun': {
        'dist_thresh': 0.12,     # Distance to temple (normalized)
        'finger_ratio': 1.2      # Index finger vs others length ratio
    },
    'smoothing': {
        'history_size': 5        # Number of frames for majority vote
    },
    'debug': {
        'enabled': False,        # Set to True to see debug overlays
        'show_roi': True,        # Show regions of interest
        'show_metrics': True     # Show numeric values
    }
}

# Display window configuration
WINDOW_WIDTH: int = 720
WINDOW_HEIGHT: int = 450
EMOJI_WINDOW_SIZE: Tuple[int, int] = (600, 600)

# MediaPipe FaceMesh landmark indices (based on canonical face model)
LEFT_EYE_IDX = [33, 159, 133, 145]    # outer, upper, inner, lower
RIGHT_EYE_IDX = [263, 386, 362, 374]
MOUTH_OUTER = [61, 291, 13, 14]       # left, right, upper, lower
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
TEMPLE_IDX = 264  # Right temple landmark

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Pattern history for temporal smoothing
PATTERN_HISTORY: Deque[Optional[int]] = deque(maxlen=CONFIG['smoothing']['history_size'])

# MediaPipe solution instances (initialized lazily in main)
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Haar Cascade classifier (initialized at module load)
eye_cascade: Optional[cv2.CascadeClassifier] = None

def _init_haar_cascade() -> Optional[cv2.CascadeClassifier]:
    """Initialize Haar Cascade classifier for eye detection."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        
        if cascade.empty():
            print("⚠️  Warning: Haar Cascade XML loaded but is empty.", file=sys.stderr)
            return None
            
        return cascade
    except Exception as e:
        print(f"⚠️  Warning: Failed to load Haar Cascade: {e}", file=sys.stderr)
        return None

eye_cascade = _init_haar_cascade()

# ============================================================================
# IMAGE ASSET LOADING
# ============================================================================

class ImageAssets:
    """Container for loaded Rocky gesture images."""
    
    def __init__(self):
        self.clip_image: Optional[np.ndarray] = None
        self.gun_image: Optional[np.ndarray] = None
        self.sunglasses_image: Optional[np.ndarray] = None
        self.tongue_image: Optional[np.ndarray] = None
        self.blank_image: Optional[np.ndarray] = None
    
    def load_from_disk(self) -> None:
        """Load and resize all Rocky gesture images."""
        image_files = {
            'clip_image': 'Rocky Pat_1.JPG',
            'gun_image': 'Rocky Pat_2.JPG',
            'sunglasses_image': 'Rocky Pat_3.JPG',
            'tongue_image': 'Rocky Pat_4.JPG'
        }
        
        for attr_name, filename in image_files.items():
            img = cv2.imread(filename)
            if img is None:
                raise FileNotFoundError(
                    f"Required image '{filename}' not found in current directory. "
                    f"Please ensure all Rocky Pat_X.JPG files are present."
                )
            
            # Resize with high-quality interpolation
            resized = cv2.resize(img, EMOJI_WINDOW_SIZE, interpolation=cv2.INTER_LANCZOS4)
            setattr(self, attr_name, resized)
        
        # Create blank/neutral image
        self.blank_image = np.zeros(
            (EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), 
            dtype=np.uint8
        )

# Global assets instance (loaded in main)
assets: Optional[ImageAssets] = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def landmark_to_pixel(lm, frame_width: int, frame_height: int) -> Tuple[int, int]:
    """Convert normalized MediaPipe landmark to pixel coordinates.
    
    Args:
        lm: MediaPipe landmark with normalized x, y coordinates [0, 1]
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        
    Returns:
        Tuple of (x, y) pixel coordinates
    """
    return int(lm.x * frame_width), int(lm.y * frame_height)


def get_landmark(face_landmarks, idx: int, w: int, h: int) -> Tuple[int, int]:
    """Extract pixel coordinates for a specific facial landmark.
    
    Args:
        face_landmarks: MediaPipe face landmark results
        idx: Landmark index (0-477 for FaceMesh)
        w: Frame width in pixels
        h: Frame height in pixels
        
    Returns:
        Tuple of (x, y) pixel coordinates
    """
    lm = face_landmarks.landmark[idx]
    return landmark_to_pixel(lm, w, h)


# ============================================================================
# PATTERN DETECTION FUNCTIONS
# ============================================================================

def eye_aspect_ratio(landmarks, eye_idx: list, w: int, h: int) -> float:
    """Calculate Eye Aspect Ratio (EAR) for eye closure detection.
    
    EAR is the ratio of vertical to horizontal eye dimensions.
    Lower values indicate more closed eyes.
    
    Reference: Soukupová and Čech (2016) - "Real-Time Eye Blink Detection"
    
    Args:
        landmarks: MediaPipe face landmarks
        eye_idx: List of 4 landmark indices [outer, upper, inner, lower]
        w: Frame width
        h: Frame height
        
    Returns:
        Eye aspect ratio (typically 0.1-0.3)
    """
    try:
        p1 = get_landmark(landmarks, eye_idx[0], w, h)  # outer corner
        p2 = get_landmark(landmarks, eye_idx[1], w, h)  # upper eyelid
        p3 = get_landmark(landmarks, eye_idx[2], w, h)  # inner corner
        p4 = get_landmark(landmarks, eye_idx[3], w, h)  # lower eyelid

        # Vertical distance (eyelid separation)
        vert = np.linalg.norm(np.array(p2) - np.array(p4))
        # Horizontal distance (eye width)
        horiz = np.linalg.norm(np.array(p1) - np.array(p3))

        if horiz == 0:
            return 0.0
        return vert / horiz
    except (IndexError, AttributeError, ZeroDivisionError):
        return 0.0


def is_one_eye_closed(face_landmarks, w: int, h: int) -> bool:
    """Detect wink gesture (one eye closed, one eye open).
    
    Pattern 1: One eye closed
    Uses Eye Aspect Ratio to detect asymmetric eye closure.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
        w: Frame width
        h: Frame height
        
    Returns:
        True if winking detected, False otherwise
    """
    try:
        left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, w, h)
        right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, w, h)

        ear_thresh = CONFIG['eye']['ear_thresh']
        diff_thresh = CONFIG['eye']['diff_thresh']

        left_closed = left_ear < ear_thresh
        right_closed = right_ear < ear_thresh

        # XOR: exactly one eye closed, with significant EAR difference
        if (left_closed ^ right_closed) and abs(left_ear - right_ear) > diff_thresh:
            return True
    except Exception:
        return False
    
    return False


def is_wearing_sunglasses(frame_bgr: np.ndarray, face_landmarks, w: int, h: int) -> bool:
    """Detect sunglasses using Haar Cascade eye detection.
    
    Pattern 3: Sunglasses
    Logic: Eyes are occluded (Haar fails to detect them) AND region is dark.
    
    Args:
        frame_bgr: Input frame in BGR color space
        face_landmarks: MediaPipe face mesh landmarks
        w: Frame width
        h: Frame height
        
    Returns:
        True if sunglasses detected, False otherwise
    """
    if eye_cascade is None:
        return False
    
    try:
        # Extract eye region bounding box
        left_eye_outer = get_landmark(face_landmarks, 33, w, h)
        right_eye_outer = get_landmark(face_landmarks, 263, w, h)
        left_eye_top = get_landmark(face_landmarks, 159, w, h)
        right_eye_bottom = get_landmark(face_landmarks, 374, w, h)
        
        # Expand ROI to capture full glasses area (40px margin)
        x1 = max(0, min(left_eye_outer[0], right_eye_outer[0]) - 40)
        x2 = min(w - 1, max(left_eye_outer[0], right_eye_outer[0]) + 40)
        y1 = max(0, min(left_eye_top[1], right_eye_bottom[1]) - 40)
        y2 = min(h - 1, max(left_eye_top[1], right_eye_bottom[1]) + 40)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        roi = frame_bgr[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Attempt to detect eyes using Haar Cascade
        eyes = eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Measure region brightness
        mean_brightness = np.mean(gray_roi)
        brightness_thresh = CONFIG['sunglasses']['brightness_thresh']
        
        # Sunglasses present if: eyes occluded (not detected) AND region is dark
        no_eyes_detected = len(eyes) == 0
        is_dark = mean_brightness < brightness_thresh
        
        return no_eyes_detected and is_dark
            
    except Exception:
        return False


def mouth_open_ratio(face_landmarks, w: int, h: int) -> float:
    """Calculate mouth aspect ratio (height/width).
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
        w: Frame width
        h: Frame height
        
    Returns:
        Mouth aspect ratio (higher = more open)
    """
    try:
        upper = get_landmark(face_landmarks, UPPER_LIP_IDX, w, h)
        lower = get_landmark(face_landmarks, LOWER_LIP_IDX, w, h)
        left = get_landmark(face_landmarks, MOUTH_OUTER[0], w, h)
        right = get_landmark(face_landmarks, MOUTH_OUTER[1], w, h)

        vert = np.linalg.norm(np.array(upper) - np.array(lower))
        horiz = np.linalg.norm(np.array(left) - np.array(right))
        
        if horiz == 0:
            return 0.0
        return vert / horiz
    except Exception:
        return 0.0


def is_tongue_out(frame_bgr: np.ndarray, face_landmarks, w: int, h: int) -> bool:
    """Detect tongue-out gesture using HSV color analysis.
    
    Pattern 4: Tongue out
    Detects if mouth is open AND contains red/pink pixels (tongue color).
    
    Args:
        frame_bgr: Input frame in BGR color space
        face_landmarks: MediaPipe face mesh landmarks
        w: Frame width
        h: Frame height
        
    Returns:
        True if tongue detected, False otherwise
    """
    try:
        # First check: mouth must be sufficiently open
        mor = mouth_open_ratio(face_landmarks, w, h)
        mouth_thresh = CONFIG['tongue']['mouth_open_thresh']
        
        if mor < mouth_thresh:
            return False

        # Extract mouth region bounding box
        xs, ys = [], []
        for idx in MOUTH_OUTER:
            x, y = get_landmark(face_landmarks, idx, w, h)
            xs.append(x)
            ys.append(y)

        x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
        y1, y2 = max(min(ys), 0), min(max(ys), h - 1)
        
        if x2 <= x1 or y2 <= y1:
            return False

        # Convert ROI to HSV for color analysis
        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define red/pink color ranges for tongue detection
        # Lower red range (0-15°)
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([15, 255, 255])
        # Upper red range (160-179°)
        lower_red2 = np.array([160, 80, 50])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Calculate percentage of red/pink pixels
        red_ratio = np.sum(mask > 0) / (mask.size + 1e-6)
        red_thresh = CONFIG['tongue']['red_ratio_thresh']

        return red_ratio > red_thresh
        
    except Exception:
        return False


def is_finger_gun_near_temple(hands_results, face_landmarks, w: int, h: int) -> bool:
    """Detect finger-gun gesture near temple.
    
    Pattern 2: Finger-gun (highest priority)
    Checks: index finger extended, thumb up, other fingers bent, hand near temple.
    
    Args:
        hands_results: MediaPipe hands detection results
        face_landmarks: MediaPipe face mesh landmarks
        w: Frame width
        h: Frame height
        
    Returns:
        True if finger-gun gesture detected, False otherwise
    """
    if not hands_results.multi_hand_landmarks:
        return False

    try:
        # Get temple position from face landmarks
        temple_x, temple_y = get_landmark(face_landmarks, TEMPLE_IDX, w, h)
        temple_vec = np.array([temple_x, temple_y])

        # Check each detected hand
        for hand_lms in hands_results.multi_hand_landmarks:
            # Extract hand landmarks (21 points per hand)
            idx_tip = hand_lms.landmark[8]    # Index finger tip
            idx_mcp = hand_lms.landmark[5]    # Index finger MCP joint
            mid_tip = hand_lms.landmark[12]   # Middle finger tip
            ring_tip = hand_lms.landmark[16]  # Ring finger tip
            pinky_tip = hand_lms.landmark[20] # Pinky tip
            thumb_tip = hand_lms.landmark[4]  # Thumb tip
            thumb_ip = hand_lms.landmark[3]   # Thumb IP joint

            # Helper: Convert normalized coords to pixels
            def to_px(lm) -> np.ndarray:
                return np.array([lm.x * w, lm.y * h])

            idx_tip_px = to_px(idx_tip)
            idx_mcp_px = to_px(idx_mcp)
            mid_tip_px = to_px(mid_tip)
            ring_tip_px = to_px(ring_tip)
            pinky_tip_px = to_px(pinky_tip)
            thumb_tip_px = to_px(thumb_tip)
            thumb_ip_px = to_px(thumb_ip)

            # Condition 1: Index finger extended (longer than other fingers)
            idx_len = np.linalg.norm(idx_tip_px - idx_mcp_px)
            mid_len = np.linalg.norm(mid_tip_px - idx_mcp_px)
            ring_len = np.linalg.norm(ring_tip_px - idx_mcp_px)
            pinky_len = np.linalg.norm(pinky_tip_px - idx_mcp_px)

            avg_other_len = np.mean([mid_len, ring_len, pinky_len])
            finger_ratio = CONFIG['gun']['finger_ratio']
            cond_index_extended = idx_len > finger_ratio * avg_other_len

            # Condition 2: Thumb up (extended)
            thumb_len = np.linalg.norm(thumb_tip_px - thumb_ip_px)
            min_thumb_len = 0.02 * max(w, h)
            cond_thumb_up = thumb_len > min_thumb_len

            # Condition 3: Hand near temple
            dist_to_temple = np.linalg.norm(idx_tip_px - temple_vec) / max(w, h)
            dist_thresh = CONFIG['gun']['dist_thresh']
            cond_near_temple = dist_to_temple < dist_thresh

            # All conditions must be met
            if cond_index_extended and cond_thumb_up and cond_near_temple:
                return True
                
    except Exception:
        return False
    
    return False


# ============================================================================
# PATTERN DETECTION ORCHESTRATOR
# ============================================================================

def detect_pattern(
    frame_bgr: np.ndarray, 
    face_results, 
    hands_results, 
    w: int, 
    h: int
) -> Optional[int]:
    """Detect active gesture pattern with priority-based decision tree.
    
    Patterns are evaluated in priority order:
    1. Pattern 2: Finger-gun (highest priority)
    2. Pattern 4: Tongue out
    3. Pattern 3: Sunglasses
    4. Pattern 1: One eye closed (lowest priority)
    
    Args:
        frame_bgr: Input frame in BGR color space
        face_results: MediaPipe face mesh results
        hands_results: MediaPipe hands results
        w: Frame width
        h: Frame height
        
    Returns:
        Pattern ID (1-4) or None if no pattern detected
    """
    if not face_results.multi_face_landmarks:
        return None

    face_landmarks = face_results.multi_face_landmarks[0]

    # Priority 1: Finger-gun (most distinctive gesture)
    if is_finger_gun_near_temple(hands_results, face_landmarks, w, h):
        return 2

    # Priority 2: Tongue out
    if is_tongue_out(frame_bgr, face_landmarks, w, h):
        return 4

    # Priority 3: Sunglasses
    if is_wearing_sunglasses(frame_bgr, face_landmarks, w, h):
        return 3

    # Priority 4: One eye closed (baseline trigger)
    if is_one_eye_closed(face_landmarks, w, h):
        return 1

    return None


def get_stable_pattern(pattern_history: Deque[Optional[int]]) -> Optional[int]:
    """Apply temporal smoothing using majority vote algorithm.
    
    Reduces flickering by requiring pattern consistency across multiple frames.
    
    Args:
        pattern_history: Deque of recent pattern detections
        
    Returns:
        Most frequent pattern in history, or None if insufficient data
    """
    if len(pattern_history) == 0:
        return None
    
    # Filter out None values
    values = [p for p in pattern_history if p is not None]
    if not values:
        return None
    
    # Return most common pattern (majority vote)
    return max(set(values), key=values.count)


# ============================================================================
# DEBUG VISUALIZATION
# ============================================================================

def draw_debug_overlay(
    frame: np.ndarray, 
    face_landmarks, 
    pattern: Optional[int], 
    stable_pattern: Optional[int], 
    w: int, 
    h: int
) -> np.ndarray:
    """Draw debug overlays showing detection regions and metrics.
    
    Args:
        frame: Input frame
        face_landmarks: MediaPipe face landmarks
        pattern: Currently detected pattern (before smoothing)
        stable_pattern: Smoothed pattern (after majority vote)
        w: Frame width
        h: Frame height
        
    Returns:
        Frame with debug overlay
    """
    if not CONFIG['debug']['enabled']:
        return frame
    
    debug_frame = frame.copy()
    
    if face_landmarks and CONFIG['debug']['show_roi']:
        # Draw eye regions (yellow)
        for eye_idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
            points = [get_landmark(face_landmarks, idx, w, h) for idx in eye_idx]
            pts = np.array(points, np.int32)
            cv2.polylines(debug_frame, [pts], True, (0, 255, 255), 1)
        
        # Draw mouth region (magenta)
        mouth_points = [get_landmark(face_landmarks, idx, w, h) for idx in MOUTH_OUTER]
        pts = np.array(mouth_points, np.int32)
        cv2.polylines(debug_frame, [pts], True, (255, 0, 255), 1)
        
        # Draw temple point (red)
        temple = get_landmark(face_landmarks, TEMPLE_IDX, w, h)
        cv2.circle(debug_frame, temple, 5, (0, 0, 255), -1)
    
    if CONFIG['debug']['show_metrics']:
        # Display detection metrics
        y_offset = 60
        cv2.putText(debug_frame, f'Raw pattern: {pattern}', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 25
        cv2.putText(debug_frame, f'Stable pattern: {stable_pattern}', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 25
        cv2.putText(debug_frame, f'History: {list(PATTERN_HISTORY)}', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return debug_frame


# ============================================================================
# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> int:
    """Main application entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    global assets
    
    # Print startup banner
    print("=" * 60)
    print("🎬 ROCKY GESTURE RECOGNITION SYSTEM v2.0")
    print("=" * 60)
    
    # Load image assets
    try:
        print("\n📂 Loading Rocky image assets...")
        assets = ImageAssets()
        assets.load_from_disk()
        print("✅ All images loaded successfully")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("\nRequired files:")
        print("  - Rocky Pat_1.JPG (one eye closed)")
        print("  - Rocky Pat_2.JPG (finger gun)")
        print("  - Rocky Pat_3.JPG (sunglasses)")
        print("  - Rocky Pat_4.JPG (tongue out)")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error loading images: {e}", file=sys.stderr)
        return 1
    
    # Initialize camera
    print("📹 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access webcam", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  1. Check camera permissions in System Settings")
        print("  2. Ensure no other app is using the camera")
        print("  3. Try changing camera index: cv2.VideoCapture(1)")
        return 1
    
    print("✅ Camera initialized")
    
    # Create display windows
    cv2.namedWindow('Rocky Output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow('Rocky Output', EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1])
    cv2.moveWindow('Camera Feed', 100, 100)
    cv2.moveWindow('Rocky Output', WINDOW_WIDTH + 150, 100)
    
    # Display usage instructions
    print("\n" + "=" * 60)
    print("📋 AVAILABLE GESTURES (Priority Order):")
    print("=" * 60)
    print("  🔫 Pat_2: FINGER-GUN → Point finger at temple (HIGHEST)")
    print("  👅 Pat_4: TONGUE OUT → Stick tongue out")
    print("  🕶️  Pat_3: SUNGLASSES → Wear dark sunglasses")
    print("  👁️  Pat_1: ONE EYE    → Close/wink one eye (BASELINE)")
    print(f"\n⚙️  Smoothing: {CONFIG['smoothing']['history_size']}-frame majority vote")
    print(f"🐛 Debug mode: {'ON' if CONFIG['debug']['enabled'] else 'OFF'}")
    print("\n⌨️  Controls:")
    print("    'q' - Quit application")
    print("    'd' - Toggle debug overlay")
    print("=" * 60 + "\n")
    
    # Initialize MediaPipe solutions
    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        ) as face_mesh, \
             mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            # Main processing loop
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Warning: Failed to grab frame", file=sys.stderr)
                    continue

                # Mirror frame horizontally for intuitive interaction
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Convert BGR to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run MediaPipe inference
                face_results = face_mesh.process(rgb)
                hands_results = hands.process(rgb)

                # Detect pattern
                pattern = detect_pattern(frame, face_results, hands_results, w, h)
                PATTERN_HISTORY.append(pattern)

                # Apply temporal smoothing
                stable_pattern = get_stable_pattern(PATTERN_HISTORY)

                # Map pattern to corresponding Rocky image
                if stable_pattern == 1:
                    rocky_img = assets.clip_image
                    state_name = "ONE_EYE_CLOSED"
                elif stable_pattern == 2:
                    rocky_img = assets.gun_image
                    state_name = "FINGER_GUN"
                elif stable_pattern == 3:
                    rocky_img = assets.sunglasses_image
                    state_name = "SUNGLASSES"
                elif stable_pattern == 4:
                    rocky_img = assets.tongue_image
                    state_name = "TONGUE_OUT"
                else:
                    rocky_img = assets.blank_image
                    state_name = "NEUTRAL"

                # Apply debug overlay if enabled
                if face_results.multi_face_landmarks:
                    display_frame = draw_debug_overlay(
                        frame, face_results.multi_face_landmarks[0],
                        pattern, stable_pattern, w, h
                    )
                else:
                    display_frame = frame

                # Prepare camera feed with status overlay
                camera_frame = cv2.resize(display_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                cv2.putText(
                    camera_frame, 
                    f'STATE: {state_name}', 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 0, 0),  # Black text
                    2, 
                    cv2.LINE_AA
                )
                cv2.putText(
                    camera_frame, 
                    'Press "q" to quit | "d" for debug', 
                    (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )

                # Display both windows
                cv2.imshow('Camera Feed', camera_frame)
                cv2.imshow('Rocky Output', rocky_img)

                # Handle keyboard input
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quit signal received")
                    break
                elif key == ord('d'):
                    CONFIG['debug']['enabled'] = not CONFIG['debug']['enabled']
                    status = 'ON' if CONFIG['debug']['enabled'] else 'OFF'
                    print(f"🐛 Debug mode: {status}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Runtime error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup resources
        print("\n🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")
    
    print("👋 Rocky Gesture Recognition terminated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
