"""
Sensor Fusion Engine for Road Accident Detection System
Combines vision, audio, and sensor data to reduce false positives
"""

import numpy as np
from typing import Tuple, Dict, Any
from enum import Enum


class AlertLevel(Enum):
    """Alert levels based on fusion score"""
    FALSE_ALARM = "FALSE_ALARM"
    WARNING = "WARNING"
    CRITICAL_CRASH = "CRITICAL_CRASH"


class SensorFusionEngine:
    """
    Fuses multiple sensor inputs to compute a unified crash probability score.
    
    Weight Distribution:
    - Vision (45%): Primary indicator from visual crash detection
    - Audio (25%): Audible crash/shattering sounds
    - G-Force (30%): Physical impact measurement
    
    This weighted approach ensures visual evidence is most important while
    requiring supporting evidence from other sensors to reduce false positives.
    """
    
    # Weight coefficients for each sensor modality
    VISION_WEIGHT = 0.45    # 45% weight to visual evidence
    AUDIO_WEIGHT = 0.25     # 25% weight to audio evidence
    G_FORCE_WEIGHT = 0.30   # 30% weight to physical impact
    
    # Decision thresholds for alert levels
    CRITICAL_THRESHOLD = 0.80    # ≥0.80: Critical crash
    WARNING_THRESHOLD = 0.50     # ≥0.50: Warning level
    # <0.50: False alarm
    
    # Audio classification constants
    AUDIO_CRASH_KEYWORDS = {"crash", "glass", "shatter", "break", "impact", "collision"}
    
    # G-Force normalization constant (5G = maximum score of 1.0)
    MAX_G_FORCE = 5.0
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the Sensor Fusion Engine.
        
        Args:
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        self.history = []  # Store fusion history for analysis
        
    def _normalize_vision_score(self, vision_confidence: float) -> float:
        """
        Normalize vision confidence score.
        
        Vision confidence comes from YOLO-based accident detection model.
        Higher values (closer to 1.0) indicate stronger visual evidence.
        
        Args:
            vision_confidence: Raw confidence from vision model (0.0-1.0)
            
        Returns:
            Normalized vision score (0.0-1.0)
        """
        # Direct mapping since vision_confidence is already normalized
        # Apply sigmoid-like transformation to emphasize high confidence
        normalized = 1 / (1 + np.exp(-10 * (vision_confidence - 0.7)))
        
        if self.verbose:
            print(f"Vision: raw={vision_confidence:.3f}, normalized={normalized:.3f}")
            
        return normalized
    
    def _normalize_audio_score(self, audio_class: str, audio_confidence: float) -> float:
        """
        Normalize audio classification score.
        
        Based on urban sound classification model detecting crash sounds.
        Only "Crash" or similar classes contribute to the score.
        
        Args:
            audio_class: Classification result string
            audio_confidence: Confidence of audio classification (0.0-1.0)
            
        Returns:
            Normalized audio score (0.0-1.0)
        """
        # Check if audio class indicates a crash event
        audio_class_lower = audio_class.lower()
        is_crash_audio = any(keyword in audio_class_lower 
                           for keyword in self.AUDIO_CRASH_KEYWORDS)
        
        # Score is 1.0 for crash sounds, 0.0 otherwise
        # Multiply by audio confidence to account for classification certainty
        audio_score = 1.0 if is_crash_audio else 0.0
        normalized = audio_score * audio_confidence
        
        if self.verbose:
            print(f"Audio: class='{audio_class}', confidence={audio_confidence:.3f}, "
                  f"is_crash={is_crash_audio}, score={normalized:.3f}")
            
        return normalized
    
    def _normalize_g_force_score(self, g_force: float) -> float:
        """
        Normalize G-force measurement.
        
        Normalizes linear acceleration to score (0.0-1.0).
        3.5G is typical threshold for significant impact.
        5.0G is capped as maximum measurable impact.
        
        Args:
            g_force: Raw G-force measurement (typically 1.0-10.0+)
            
        Returns:
            Normalized G-force score (0.0-1.0)
        """
        # Normalize with cap at MAX_G_FORCE
        # Formula: min(g_force / MAX_G_FORCE, 1.0)
        raw_score = g_force / self.MAX_G_FORCE
        normalized = min(raw_score, 1.0)
        
        # Apply non-linear scaling to emphasize high G-forces
        # This gives more weight to severe impacts
        normalized = np.power(normalized, 0.8)  # Square root-like transformation
        
        if self.verbose:
            print(f"G-Force: raw={g_force:.3f}, normalized={normalized:.3f}")
            
        return normalized
    
    def compute_fusion_score(self, 
                           vision_confidence: float,
                           audio_class: str,
                           audio_confidence: float,
                           g_force: float) -> float:
        """
        Compute weighted fusion score using all sensor inputs.
        
        Fusion Formula:
        Fusion_Score = (Vision_Score * 0.45) + 
                      (Audio_Score * 0.25) + 
                      (G_Force_Score * 0.30)
        
        Args:
            vision_confidence: Confidence from vision model (0.0-1.0)
            audio_class: Audio classification string
            audio_confidence: Audio classification confidence (0.0-1.0)
            g_force: G-force measurement
            
        Returns:
            Fusion score (0.0-1.0)
        """
        # Step 1: Normalize individual sensor scores
        vision_score = self._normalize_vision_score(vision_confidence)
        audio_score = self._normalize_audio_score(audio_class, audio_confidence)
        g_force_score = self._normalize_g_force_score(g_force)
        
        # Step 2: Apply weighted fusion
        fusion_score = (
            vision_score * self.VISION_WEIGHT +
            audio_score * self.AUDIO_WEIGHT +
            g_force_score * self.G_FORCE_WEIGHT
        )
        
        # Ensure score is within bounds
        fusion_score = max(0.0, min(1.0, fusion_score))
        
        # Store in history for trend analysis
        self.history.append({
            'vision': vision_score,
            'audio': audio_score,
            'g_force': g_force_score,
            'fusion': fusion_score,
            'raw_inputs': {
                'vision_confidence': vision_confidence,
                'audio_class': audio_class,
                'audio_confidence': audio_confidence,
                'g_force': g_force
            }
        })
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        if self.verbose:
            print(f"\nFusion Breakdown:")
            print(f"  Vision: {vision_score:.3f} × {self.VISION_WEIGHT} = {vision_score * self.VISION_WEIGHT:.3f}")
            print(f"  Audio:  {audio_score:.3f} × {self.AUDIO_WEIGHT} = {audio_score * self.AUDIO_WEIGHT:.3f}")
            print(f"  G-Force: {g_force_score:.3f} × {self.G_FORCE_WEIGHT} = {g_force_score * self.G_FORCE_WEIGHT:.3f}")
            print(f"  Total: {fusion_score:.3f}")
        
        return fusion_score
    
    def determine_alert_level(self, fusion_score: float) -> AlertLevel:
        """
        Determine alert level based on fusion score threshold.
        
        Threshold Logic:
        - Score ≥ 0.80: CRITICAL_CRASH (High confidence accident)
        - Score ≥ 0.50: WARNING (Possible accident, needs verification)
        - Score < 0.50: FALSE_ALARM (Likely false positive)
        
        Args:
            fusion_score: Computed fusion score (0.0-1.0)
            
        Returns:
            AlertLevel enum value
        """
        if fusion_score >= self.CRITICAL_THRESHOLD:
            return AlertLevel.CRITICAL_CRASH
        elif fusion_score >= self.WARNING_THRESHOLD:
            return AlertLevel.WARNING
        else:
            return AlertLevel.FALSE_ALARM
    
    def analyze(self,
               vision_confidence: float,
               audio_class: str,
               audio_confidence: float,
               g_force: float) -> Dict[str, Any]:
        """
        Main analysis method that combines all sensor data.
        
        Args:
            vision_confidence: Confidence from vision model (0.0-1.0)
            audio_class: Audio classification string
            audio_confidence: Audio classification confidence (0.0-1.0)
            g_force: G-force measurement
            
        Returns:
            Dictionary containing:
            - status: AlertLevel as string
            - score: Fusion score (0.0-1.0)
            - breakdown: Individual sensor scores
            - recommendation: Action recommendation
        """
        # Compute fusion score
        fusion_score = self.compute_fusion_score(
            vision_confidence, audio_class, audio_confidence, g_force
        )
        
        # Determine alert level
        alert_level = self.determine_alert_level(fusion_score)
        
        # Prepare response
        response = {
            'status': alert_level.value,
            'score': round(fusion_score, 3),
            'breakdown': {
                'vision': round(self._normalize_vision_score(vision_confidence), 3),
                'audio': round(self._normalize_audio_score(audio_class, audio_confidence), 3),
                'g_force': round(self._normalize_g_force_score(g_force), 3),
                'weights': {
                    'vision': self.VISION_WEIGHT,
                    'audio': self.AUDIO_WEIGHT,
                    'g_force': self.G_FORCE_WEIGHT
                }
            },
            'thresholds': {
                'critical': self.CRITICAL_THRESHOLD,
                'warning': self.WARNING_THRESHOLD
            }
        }
        
        # Add recommendations based on alert level
        if alert_level == AlertLevel.CRITICAL_CRASH:
            response['recommendation'] = 'TRIGGER_EMERGENCY_ALERT'
            response['message'] = 'High-confidence accident detected. Immediate response required.'
        elif alert_level == AlertLevel.WARNING:
            response['recommendation'] = 'REQUEST_DRIVER_STATUS'
            response['message'] = 'Possible accident detected. Verification needed.'
        else:
            response['recommendation'] = 'IGNORE'
            response['message'] = 'Likely false alarm. No action required.'
        
        return response
    
    def get_fusion_history(self, limit: int = 10) -> list:
        """
        Get recent fusion history for analysis.
        
        Args:
            limit: Number of recent entries to return
            
        Returns:
            List of recent fusion entries
        """
        return self.history[-limit:] if self.history else []


# Example usage and testing
if __name__ == "__main__":
    # Initialize fusion engine with verbose output
    fusion_engine = SensorFusionEngine(verbose=True)
    
    # Test Case 1: Clear Crash Scenario (All sensors positive)
    print("\n" + "="*60)
    print("TEST CASE 1: CLEAR CRASH SCENARIO")
    print("="*60)
    result1 = fusion_engine.analyze(
        vision_confidence=0.95,  # High visual confidence
        audio_class="Crash",     # Crash audio detected
        audio_confidence=0.90,   # High audio confidence
        g_force=4.8             # High G-force impact
    )
    print(f"Result: {result1['status']} (Score: {result1['score']})")
    print(f"Recommendation: {result1['recommendation']}")
    
    # Test Case 2: Marginal Scenario (Mixed signals)
    print("\n" + "="*60)
    print("TEST CASE 2: MARGINAL SCENARIO")
    print("="*60)
    result2 = fusion_engine.analyze(
        vision_confidence=0.65,  # Moderate visual confidence
        audio_class="Siren",     # Siren, not crash
        audio_confidence=0.80,   # High confidence but wrong class
        g_force=2.5             # Low G-force
    )
    print(f"Result: {result2['status']} (Score: {result2['score']})")
    print(f"Recommendation: {result2['recommendation']}")
    
    # Test Case 3: False Alarm (Low scores)
    print("\n" + "="*60)
    print("TEST CASE 3: FALSE ALARM")
    print("="*60)
    result3 = fusion_engine.analyze(
        vision_confidence=0.30,  # Low visual confidence
        audio_class="Normal",    # Normal sounds
        audio_confidence=0.95,   # High confidence but normal class
        g_force=1.2             # Normal driving G-force
    )
    print(f"Result: {result3['status']} (Score: {result3['score']})")
    print(f"Recommendation: {result3['recommendation']}")
    
    # Show fusion history
    print("\n" + "="*60)
    print("FUSION HISTORY (Last 3 entries)")
    print("="*60)
    for entry in fusion_engine.get_fusion_history(3):
        print(f"Fusion Score: {entry['fusion']:.3f}")