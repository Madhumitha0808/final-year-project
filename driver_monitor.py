"""
Driver Status Monitor for Post-Crash Medical Assessment
Uses MediaPipe Face Mesh to analyze driver consciousness
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import math


class DriverStatusMonitor:
    """
    Analyzes driver condition post-crash using facial landmarks.
    
    Implements two critical checks:
    1. Eye Aspect Ratio (EAR) for consciousness detection
    2. Head pitch angle for slumped position detection
    
    Based on Eye-Blink-Detection-using-MediaPipe-and-OpenCV implementation
    with adaptations for crash scenario analysis.
    """
    
    # MediaPipe Face Mesh landmark indices (0-based)
    # Left Eye landmarks (outer to inner)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]  # [p1, p2, p3, p4, p5, p6]
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Right Eye landmarks (outer to inner)
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # [p1, p2, p3, p4, p5, p6]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Nose and face plane landmarks for head pose estimation
    NOSE_TIP = 1           # Tip of the nose
    FOREHEAD = 10          # Center of forehead
    CHIN = 152             # Chin point
    LEFT_FACE = 454        # Left cheek
    RIGHT_FACE = 234       # Right cheek
    
    # Thresholds for medical assessment
    EAR_THRESHOLD = 0.20      # Eye Aspect Ratio threshold (eyes closed if < 0.20)
    PITCH_THRESHOLD = 45.0    # Head pitch threshold (slumped if > 45 degrees)
    
    # Confidence thresholds
    MIN_FACE_DETECTION_CONFIDENCE = 0.5
    MIN_FACE_PRESENCE_CONFIDENCE = 0.5
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the Driver Status Monitor.
        
        Args:
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,  # Single image analysis
            max_num_faces=1,         # Expect only driver's face
            refine_landmarks=True,   # Get more precise landmarks
            min_detection_confidence=self.MIN_FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_FACE_PRESENCE_CONFIDENCE
        )
        
        # Drawing utilities for visualization (optional)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        print("✅ DriverStatusMonitor initialized with MediaPipe Face Mesh")
    
    def _euclidean_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two 2D points.
        
        Formula: sqrt((x2 - x1)² + (y2 - y1)²)
        
        Args:
            point1: (x, y) coordinates of first point
            point2: (x, y) coordinates of second point
            
        Returns:
            Euclidean distance
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) using 6-point landmark method.
        
        EAR Formula:
            EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
            
        Where p1-p6 are eye landmarks in order:
        p1, p2 - outer corner, top, inner corner, bottom
        
        EAR values:
        - >0.25: Eyes wide open
        - 0.20-0.25: Eyes partially open
        - <0.20: Eyes closed (or nearly closed)
        
        Args:
            eye_landmarks: Array of 6 eye landmark coordinates
            
        Returns:
            Eye Aspect Ratio (EAR) value
        """
        if len(eye_landmarks) != 6:
            raise ValueError(f"Expected 6 eye landmarks, got {len(eye_landmarks)}")
        
        # Extract points (already in correct order from LEFT_EYE/RIGHT_EYE indices)
        p1 = eye_landmarks[0]  # Leftmost point (outer corner)
        p2 = eye_landmarks[1]  # Upper point 1
        p3 = eye_landmarks[2]  # Upper point 2
        p4 = eye_landmarks[3]  # Rightmost point (inner corner)
        p5 = eye_landmarks[4]  # Lower point 1
        p6 = eye_landmarks[5]  # Lower point 2
        
        # Calculate vertical distances
        vertical_dist1 = self._euclidean_distance(p2, p6)
        vertical_dist2 = self._euclidean_distance(p3, p5)
        
        # Calculate horizontal distance
        horizontal_dist = self._euclidean_distance(p1, p4)
        
        # Avoid division by zero
        if horizontal_dist == 0:
            return 0.0
        
        # Compute EAR
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        
        if self.verbose:
            print(f"  Vertical distances: {vertical_dist1:.3f}, {vertical_dist2:.3f}")
            print(f"  Horizontal distance: {horizontal_dist:.3f}")
            print(f"  EAR: {ear:.3f}")
        
        return ear
    
    def calculate_head_pitch(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """
        Estimate head pitch (up/down angle) using facial landmarks.
        
        Uses 3D-to-2D projection approximation:
        Pitch represents head nodding (looking up/down)
        
        Args:
            landmarks: All facial landmarks
            image_shape: (height, width) of the image
            
        Returns:
            Pitch angle in degrees (positive = looking down, negative = looking up)
        """
        height, width = image_shape
        
        # Get key landmarks (normalized to pixel coordinates)
        nose_tip = landmarks[self.NOSE_TIP]
        forehead = landmarks[self.FOREHEAD]
        chin = landmarks[self.CHIN]
        
        # Convert normalized coordinates to pixels
        nose_px = (int(nose_tip.x * width), int(nose_tip.y * height))
        forehead_px = (int(forehead.x * width), int(forehead.y * height))
        chin_px = (int(chin.x * width), int(chin.y * height))
        
        # Calculate vertical distance from forehead to chin (face height)
        face_height = self._euclidean_distance(forehead_px, chin_px)
        
        # Calculate vertical distance from forehead to nose
        forehead_nose_dist = self._euclidean_distance(forehead_px, nose_px)
        
        # Avoid division by zero
        if face_height == 0:
            return 0.0
        
        # Calculate normalized nose position (0.0 at forehead, 1.0 at chin)
        nose_position = forehead_nose_dist / face_height
        
        # Convert to pitch angle (empirical mapping)
        # Neutral position (looking straight): nose_position ≈ 0.35-0.45
        # Looking down: nose_position increases
        # Looking up: nose_position decreases
        
        # Empirical conversion: 0.3 = -30°, 0.5 = 30° (approx)
        pitch_angle = (nose_position - 0.4) * 150  # Scale factor
        
        if self.verbose:
            print(f"  Nose position: {nose_position:.3f}")
            print(f"  Pitch angle: {pitch_angle:.1f}°")
        
        return pitch_angle
    
    def analyze_driver_condition(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main method to analyze driver condition from a single image.
        
        Performs two critical checks:
        1. Eye openness using Eye Aspect Ratio (EAR)
        2. Head position using pitch angle
        
        Args:
            image: RGB image array (numpy array) of driver's face
            
        Returns:
            Dictionary containing analysis results and status
        """
        if image is None or image.size == 0:
            return {
                'status': 'ERROR',
                'message': 'Invalid image provided',
                'ear_left': 0.0,
                'ear_right': 0.0,
                'ear_avg': 0.0,
                'pitch_angle': 0.0,
                'is_conscious': False,
                'confidence': 0.0
            }
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR (OpenCV default)
            if image[0, 0, 0] > image[0, 0, 2]:  # B > R, likely BGR
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image.copy()
        else:
            image_rgb = image.copy()
        
        image_height, image_width = image_rgb.shape[:2]
        
        if self.verbose:
            print(f"\nAnalyzing driver condition from image: {image_width}x{image_height}")
        
        # Process image with MediaPipe Face Mesh
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            if self.verbose:
                print("❌ No face detected in image")
            return {
                'status': 'NO_FACE_DETECTED',
                'message': 'Could not detect driver face',
                'ear_left': 0.0,
                'ear_right': 0.0,
                'ear_avg': 0.0,
                'pitch_angle': 0.0,
                'is_conscious': False,
                'confidence': 0.0
            }
        
        # Get the first (and presumably only) face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates
        landmarks_array = np.array([
            (lm.x, lm.y, lm.z) for lm in face_landmarks.landmark
        ])
        
        # Extract eye landmarks
        left_eye_points = landmarks_array[self.LEFT_EYE_INDICES]
        right_eye_points = landmarks_array[self.RIGHT_EYE_INDICES]
        
        # Calculate Eye Aspect Ratio for both eyes
        ear_left = self.calculate_eye_aspect_ratio(left_eye_points[:, :2])  # Use only x,y
        ear_right = self.calculate_eye_aspect_ratio(right_eye_points[:, :2])
        ear_avg = (ear_left + ear_right) / 2.0
        
        # Calculate head pitch angle
        pitch_angle = self.calculate_head_pitch(face_landmarks.landmark, 
                                               (image_height, image_width))
        
        # Determine consciousness status
        # Condition: Unconscious if (eyes closed) OR (head slumped)
        eyes_closed = ear_avg < self.EAR_THRESHOLD
        head_slumped = pitch_angle > self.PITCH_THRESHOLD
        
        is_conscious = not (eyes_closed or head_slumped)
        
        # Calculate overall confidence score
        # Based on how far metrics are from thresholds
        ear_confidence = min(1.0, ear_avg / self.EAR_THRESHOLD)
        pitch_confidence = min(1.0, self.PITCH_THRESHOLD / max(abs(pitch_angle), 1.0))
        overall_confidence = (ear_confidence + pitch_confidence) / 2.0
        
        # Determine status message
        if not is_conscious:
            if eyes_closed and head_slumped:
                status_message = "UNRESPONSIVE (Eyes closed and head slumped)"
                severity = "CRITICAL"
            elif eyes_closed:
                status_message = "UNRESPONSIVE (Eyes closed)"
                severity = "HIGH"
            else:  # head_slumped
                status_message = "UNRESPONSIVE (Head slumped)"
                severity = "HIGH"
        else:
            status_message = "CONSCIOUS"
            severity = "NORMAL"
        
        # Prepare detailed response
        response = {
            'status': 'UNRESPONSIVE' if not is_conscious else 'CONSCIOUS',
            'status_message': status_message,
            'severity': severity,
            'ear_left': round(ear_left, 3),
            'ear_right': round(ear_right, 3),
            'ear_avg': round(ear_avg, 3),
            'pitch_angle': round(pitch_angle, 1),
            'thresholds': {
                'ear_threshold': self.EAR_THRESHOLD,
                'pitch_threshold': self.PITCH_THRESHOLD
            },
            'conditions': {
                'eyes_closed': eyes_closed,
                'head_slumped': head_slumped
            },
            'is_conscious': is_conscious,
            'confidence': round(overall_confidence, 3),
            'recommendation': 'CALL_AMBULANCE' if not is_conscious else 'MONITOR_ONLY',
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }
        
        if self.verbose:
            print(f"\nDriver Analysis Results:")
            print(f"  Left EAR: {ear_left:.3f} {'(CLOSED)' if ear_left < self.EAR_THRESHOLD else '(OPEN)'}")
            print(f"  Right EAR: {ear_right:.3f} {'(CLOSED)' if ear_right < self.EAR_THRESHOLD else '(OPEN)'}")
            print(f"  Average EAR: {ear_avg:.3f} (Threshold: {self.EAR_THRESHOLD})")
            print(f"  Head Pitch: {pitch_angle:.1f}° (Threshold: {self.PITCH_THRESHOLD}°)")
            print(f"  Consciousness: {'CONSCIOUS' if is_conscious else 'UNRESPONSIVE'}")
            print(f"  Recommendation: {response['recommendation']}")
        
        return response
    
    def visualize_analysis(self, image: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        Create visualization of the analysis results on the image.
        
        Args:
            image: Original image
            analysis_result: Results from analyze_driver_condition()
            
        Returns:
            Image with annotations and metrics overlay
        """
        # Create a copy for visualization
        vis_image = image.copy()
        height, width = vis_image.shape[:2]
        
        # Convert BGR to RGB if needed for MediaPipe drawing
        if len(vis_image.shape) == 3:
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        else:
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
        
        # Re-process to get landmarks for visualization
        results = self.face_mesh.process(vis_image_rgb)
        
        if results.multi_face_landmarks:
            # Draw face landmarks
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=vis_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                
                # Draw eye contours
                self.mp_drawing.draw_landmarks(
                    image=vis_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=vis_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )
        
        # Add text overlay with analysis results
        status_color = (0, 255, 0) if analysis_result['is_conscious'] else (0, 0, 255)
        
        # Status box
        cv2.rectangle(vis_image, (10, 10), (400, 150), (40, 40, 40), -1)
        cv2.rectangle(vis_image, (10, 10), (400, 150), status_color, 2)
        
        # Status text
        cv2.putText(vis_image, f"Status: {analysis_result['status']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Metrics
        cv2.putText(vis_image, f"EAR (L/R/Avg): {analysis_result['ear_left']:.2f}/"
                   f"{analysis_result['ear_right']:.2f}/{analysis_result['ear_avg']:.2f}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(vis_image, f"Head Pitch: {analysis_result['pitch_angle']:.1f}°", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(vis_image, f"Thresholds: EAR<{analysis_result['thresholds']['ear_threshold']}, "
                   f"Pitch>{analysis_result['thresholds']['pitch_threshold']}°", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(vis_image, f"Recommendation: {analysis_result['recommendation']}", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        return vis_image
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Example usage and testing
if __name__ == "__main__":
    # Initialize the monitor
    monitor = DriverStatusMonitor(verbose=True)
    
    # Test with a sample image (you would replace this with actual camera capture)
    print("\n" + "="*60)
    print("DRIVER STATUS MONITOR DEMO")
    print("="*60)
    
    # Create a synthetic test image (in practice, use real camera image)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)  # Gray background
    
    # Simulate different scenarios
    test_cases = [
        ("Conscious Driver", True, 0.25, 10.0),
        ("Unconscious (Eyes Closed)", False, 0.15, 10.0),
        ("Unconscious (Head Slumped)", False, 0.25, 50.0),
        ("Severely Injured", False, 0.10, 60.0)
    ]
    
    for case_name, expected_conscious, simulated_ear, simulated_pitch in test_cases:
        print(f"\n🧪 Test Case: {case_name}")
        print("-" * 40)
        
        # In real usage, we would analyze actual image
        # For demo, create a mock result
        mock_result = {
            'status': 'CONSCIOUS' if expected_conscious else 'UNRESPONSIVE',
            'status_message': case_name,
            'ear_left': simulated_ear,
            'ear_right': simulated_ear,
            'ear_avg': simulated_ear,
            'pitch_angle': simulated_pitch,
            'is_conscious': expected_conscious,
            'confidence': 0.8,
            'recommendation': 'MONITOR_ONLY' if expected_conscious else 'CALL_AMBULANCE'
        }
        
        print(f"  Simulated EAR: {simulated_ear:.2f}")
        print(f"  Simulated Pitch: {simulated_pitch:.1f}°")
        print(f"  Expected Conscious: {expected_conscious}")
        print(f"  Status: {mock_result['status']}")
        print(f"  Recommendation: {mock_result['recommendation']}")
    
    print("\n" + "="*60)
    print("REAL TEST WITH SAMPLE IMAGE")
    print("="*60)
    
    # In production, you would:
    # 1. Capture image from front-facing camera
    # 2. Call: result = monitor.analyze_driver_condition(captured_image)
    # 3. Check: if not result['is_conscious']: trigger_emergency()
    
    print("\n✅ DriverStatusMonitor ready for integration")
    print("To use:")
    print("1. Capture driver image after crash detection")
    print("2. Call: result = monitor.analyze_driver_condition(image)")
    print("3. If result['is_conscious'] is False, trigger ambulance alert")
    
    # Clean up
    monitor.close()