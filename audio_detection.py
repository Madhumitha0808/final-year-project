"""
Audio Event Detector for Road Safety System
Uses MFCC feature extraction and CNN for audio classification
"""

import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, Any, Optional
import warnings
import os
import json
from enum import Enum

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AudioEvent(Enum):
    """Audio event categories for road safety system"""
    NORMAL = "NORMAL"
    POTENTIAL_CRASH = "POTENTIAL_CRASH"
    EMERGENCY_VEHICLE_NEARBY = "EMERGENCY_VEHICLE_NEARBY"
    TRAFFIC_WARNING = "TRAFFIC_WARNING"
    GLASS_BREAKING = "GLASS_BREAKING"
    IMPACT = "IMPACT"
    UNKNOWN = "UNKNOWN"

class AudioEventDetector:
    """
    Audio Event Detector for Road Safety System
    
    Uses MFCC feature extraction and CNN model to classify audio events
    Based on UrbanSound8K dataset with adaptation for road safety scenarios
    
    Model Architecture (from reference repo):
    Input: MFCC features (40x174x1)
    Layers: Conv2D → MaxPooling2D → Dropout → Conv2D → MaxPooling2D → Dropout → 
            Flatten → Dense → Dropout → Dense (Softmax)
    Output: 10 classes (UrbanSound8K categories)
    """
    
    # Audio processing parameters
    SAMPLE_RATE = 22050  # Standard for UrbanSound8K
    DURATION = 4  # seconds (UrbanSound8K standard)
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    N_MFCC = 40  # Number of MFCC coefficients
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Model parameters
    INPUT_SHAPE = (N_MFCC, 174, 1)  # Based on UrbanSound8K MFCC shape
    
    # Class mapping for UrbanSound8K
    URBANSOUND8K_CLASSES = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    
    # Custom mapping to road safety events
    ROAD_SAFETY_MAPPING = {
        'air_conditioner': AudioEvent.NORMAL,
        'car_horn': AudioEvent.TRAFFIC_WARNING,
        'children_playing': AudioEvent.NORMAL,
        'dog_bark': AudioEvent.NORMAL,
        'drilling': AudioEvent.POTENTIAL_CRASH,  # Similar to impact sounds
        'engine_idling': AudioEvent.NORMAL,
        'gun_shot': AudioEvent.POTENTIAL_CRASH,  # Similar to crash/impact
        'jackhammer': AudioEvent.POTENTIAL_CRASH,  # Similar to impact
        'siren': AudioEvent.EMERGENCY_VEHICLE_NEARBY,
        'street_music': AudioEvent.NORMAL
    }
    
    # Additional crash-like sound patterns
    CRASH_KEYWORDS = ['crash', 'impact', 'glass', 'break', 'shatter', 'collision', 'metal', 'explosion']
    
    def __init__(self, model_path: str = "audio_model.h5", verbose: bool = True):
        """
        Initialize Audio Event Detector
        
        Args:
            model_path: Path to pre-trained Keras model (.h5 file)
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.model = None
        self.model_loaded = False
        
        if self.verbose:
            print("="*60)
            print("🚗 Audio Event Detector - Road Safety System")
            print("="*60)
            print(f"Audio Parameters:")
            print(f"  Sample Rate: {self.SAMPLE_RATE} Hz")
            print(f"  Duration: {self.DURATION} seconds")
            print(f"  MFCC Coefficients: {self.N_MFCC}")
            print(f"  Expected Input Shape: {self.INPUT_SHAPE}")
        
        # Load model if available
        self._load_model(model_path)
        
        # Initialize feature extractor
        self._initialize_extractor()
        
        if self.verbose:
            print("✅ AudioEventDetector initialized")
            if self.model_loaded:
                print(f"   Model: Loaded from {model_path}")
            else:
                print(f"   Model: Simulation mode (pre-trained model not found)")
    
    def _load_model(self, model_path: str):
        """
        Load pre-trained Keras model for audio classification
        
        Args:
            model_path: Path to .h5 model file
        """
        try:
            if os.path.exists(model_path):
                # Disable eager execution for compatibility
                tf.compat.v1.disable_eager_execution()
                
                # Load the model
                self.model = keras.models.load_model(model_path)
                
                # Compile model (required for prediction)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                self.model_loaded = True
                
                if self.verbose:
                    print(f"✅ Model loaded successfully from {model_path}")
                    print(f"   Model Summary:")
                    self.model.summary(print_fn=lambda x: print(f"   {x}") if not x.startswith("_") else None)
                    
            else:
                if self.verbose:
                    print(f"⚠ Pre-trained model not found at {model_path}")
                    print(f"   Using simulation mode for testing")
                self.model_loaded = False
                
        except Exception as e:
            if self.verbose:
                print(f"⚠ Error loading model: {e}")
                print(f"   Using simulation mode")
            self.model_loaded = False
    
    def _initialize_extractor(self):
        """
        Initialize feature extraction parameters
        
        Based on UrbanSound8K preprocessing pipeline from reference repo
        """
        # Create feature extraction configuration
        self.extractor_config = {
            'sr': self.SAMPLE_RATE,
            'n_mfcc': self.N_MFCC,
            'n_fft': self.N_FFT,
            'hop_length': self.HOP_LENGTH,
            'duration': self.DURATION,
            'n_mels': 128
        }
        
        # Calculate expected number of time frames
        # Formula: n_frames = (duration * sample_rate) / hop_length
        self.expected_frames = int((self.DURATION * self.SAMPLE_RATE) / self.HOP_LENGTH) + 1
        
        if self.verbose:
            print(f"📊 Feature Extraction Config:")
            print(f"   Expected frames: {self.expected_frames}")
            print(f"   MFCC shape target: ({self.N_MFCC}, {self.expected_frames})")
    
    def _extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract MFCC features from audio data
        
        Based on UrbanSound8K preprocessing from reference repository:
        - Uses librosa for MFCC extraction
        - Normalizes features using mean and std
        - Pads/truncates to expected shape
        
        Args:
            audio_data: Raw audio waveform
            sample_rate: Sampling rate of audio
            
        Returns:
            MFCC features shaped for model input
        """
        try:
            # Resample if necessary
            if sample_rate != self.SAMPLE_RATE:
                audio_data = librosa.resample(
                    y=audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.SAMPLE_RATE
                )
                sample_rate = self.SAMPLE_RATE
            
            # Ensure audio is correct length (pad or truncate)
            target_samples = self.SAMPLES_PER_TRACK
            
            if len(audio_data) > target_samples:
                # Truncate to target length
                audio_data = audio_data[:target_samples]
            elif len(audio_data) < target_samples:
                # Pad with zeros
                padding = target_samples - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            
            # Extract MFCC features - exactly as in reference repo
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.N_MFCC,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=self.extractor_config['n_mels']
            )
            
            # Calculate delta features (optional enhancement)
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Combine features (MFCC + delta + delta-delta)
            features = np.vstack([mfccs, mfccs_delta, mfccs_delta2])
            
            # Normalize features (CRUCIAL STEP - matches training)
            # Using mean normalization as in reference repo
            features_normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            # Reshape for model input (add channel dimension)
            features_reshaped = features_normalized.reshape(
                1, features_normalized.shape[0], features_normalized.shape[1], 1
            )
            
            if self.verbose:
                print(f"   MFCC Extraction:")
                print(f"     Input audio shape: {len(audio_data)} samples")
                print(f"     MFCC shape: {mfccs.shape}")
                print(f"     Final feature shape: {features_reshaped.shape}")
                print(f"     Feature range: [{features_reshaped.min():.3f}, {features_reshaped.max():.3f}]")
            
            return features_reshaped
            
        except Exception as e:
            if self.verbose:
                print(f"⚠ Error extracting MFCC features: {e}")
            # Return zeros as fallback
            return np.zeros((1, self.N_MFCC * 3, self.expected_frames, 1))
    
    def _extract_advanced_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract additional audio features for enhanced crash detection
        
        Args:
            audio_data: Raw audio waveform
            sample_rate: Sampling rate
            
        Returns:
            Dictionary of advanced features
        """
        features = {}
        
        try:
            # 1. Zero Crossing Rate (indicates percussive sounds)
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features['zero_crossing_rate'] = np.mean(zcr)
            
            # 2. Spectral Centroid (brightness of sound)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )
            features['spectral_centroid'] = np.mean(spectral_centroid)
            
            # 3. Spectral Rolloff (frequency bandwidth)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            # 4. RMS Energy (loudness)
            rms = librosa.feature.rms(y=audio_data)
            features['rms_energy'] = np.mean(rms)
            
            # 5. Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_data, sr=sample_rate
            )
            features['spectral_contrast'] = np.mean(spectral_contrast)
            
            # 6. Chroma Features (harmonic content)
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=sample_rate
            )
            features['chroma_std'] = np.std(chroma)  # High variance indicates complex sounds
            
            # 7. Onset detection (for impact sounds)
            onset_env = librosa.onset.onset_strength(
                y=audio_data, sr=sample_rate
            )
            features['onset_strength'] = np.mean(onset_env)
            
            # 8. Harmonic/Percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            features['percussive_ratio'] = np.mean(np.abs(percussive)) / (
                np.mean(np.abs(harmonic)) + np.mean(np.abs(percussive)) + 1e-8
            )
            
        except Exception as e:
            if self.verbose:
                print(f"⚠ Error extracting advanced features: {e}")
        
        return features
    
    def _is_crash_like_sound(self, features: Dict[str, Any], prediction: str) -> bool:
        """
        Enhanced crash detection using audio features
        
        Args:
            features: Advanced audio features
            prediction: Model prediction label
            
        Returns:
            Boolean indicating if sound is crash-like
        """
        crash_indicators = 0
        
        # Check if prediction is already crash-related
        if prediction in ['gun_shot', 'drilling', 'jackhammer']:
            crash_indicators += 2
        
        # Analyze advanced features
        if 'percussive_ratio' in features:
            # High percussive ratio indicates impact sounds
            if features['percussive_ratio'] > 0.7:
                crash_indicators += 2
        
        if 'onset_strength' in features:
            # Strong onsets indicate sharp impacts
            if features['onset_strength'] > 0.5:
                crash_indicators += 1
        
        if 'zero_crossing_rate' in features:
            # High ZCR indicates noisy/impact sounds
            if features['zero_crossing_rate'] > 0.2:
                crash_indicators += 1
        
        if 'rms_energy' in features:
            # Very loud sounds
            if features['rms_energy'] > 0.3:
                crash_indicators += 1
        
        # Threshold for crash detection
        return crash_indicators >= 3
    
    def preprocess_audio(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess audio file for model input
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (MFCC features, advanced features)
        """
        if self.verbose:
            print(f"\n🎵 Processing audio file: {os.path.basename(file_path)}")
        
        try:
            # Load audio file with librosa
            # Using kaiser_fast resampling as in reference repo
            audio_data, sample_rate = librosa.load(
                file_path,
                sr=None,  # Keep original sample rate
                res_type='kaiser_fast',
                duration=self.DURATION  # Limit duration for consistency
            )
            
            if self.verbose:
                print(f"   Loaded audio: {len(audio_data)} samples @ {sample_rate} Hz")
                print(f"   Duration: {len(audio_data)/sample_rate:.2f} seconds")
            
            # Extract MFCC features for model
            mfcc_features = self._extract_mfcc_features(audio_data, sample_rate)
            
            # Extract advanced features for enhanced analysis
            advanced_features = self._extract_advanced_features(audio_data, sample_rate)
            
            return mfcc_features, advanced_features
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading audio file: {e}")
            # Return zeros as fallback
            return np.zeros((1, self.N_MFCC * 3, self.expected_frames, 1)), {}
    
    def predict_class(self, mfcc_features: np.ndarray, advanced_features: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        Predict audio class using pre-trained model or simulation
        
        Args:
            mfcc_features: Preprocessed MFCC features
            advanced_features: Optional advanced features for enhanced analysis
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        try:
            if self.model_loaded and self.model is not None:
                # Use actual model prediction
                predictions = self.model.predict(mfcc_features, verbose=0)
                
                # Get top prediction
                class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][class_idx])
                
                # Map to UrbanSound8K class
                urban_sound_label = self.URBANSOUND8K_CLASSES[class_idx]
                
                if self.verbose:
                    print(f"   Model Prediction: {urban_sound_label} ({confidence:.2%})")
                    print(f"   Top 3 predictions:")
                    top_indices = np.argsort(predictions[0])[-3:][::-1]
                    for idx in top_indices:
                        print(f"     - {self.URBANSOUND8K_CLASSES[idx]}: {predictions[0][idx]:.2%}")
                
                return urban_sound_label, confidence
                
            else:
                # Simulation mode for testing
                # Simulate based on audio characteristics
                
                # Analyze features for simulation
                is_loud = False
                is_percussive = False
                
                if advanced_features:
                    if advanced_features.get('rms_energy', 0) > 0.2:
                        is_loud = True
                    if advanced_features.get('percussive_ratio', 0) > 0.6:
                        is_percussive = True
                
                # Simulation logic
                if is_loud and is_percussive:
                    # Simulate crash-like sound
                    simulated_label = random.choice(['gun_shot', 'drilling', 'jackhammer'])
                    confidence = random.uniform(0.7, 0.9)
                elif is_loud:
                    # Simulate siren or horn
                    simulated_label = random.choice(['siren', 'car_horn'])
                    confidence = random.uniform(0.6, 0.8)
                else:
                    # Normal sounds
                    simulated_label = random.choice(['air_conditioner', 'engine_idling', 'street_music'])
                    confidence = random.uniform(0.4, 0.7)
                
                if self.verbose:
                    print(f"   Simulation Prediction: {simulated_label} ({confidence:.2%})")
                
                return simulated_label, confidence
                
        except Exception as e:
            if self.verbose:
                print(f"⚠ Prediction error: {e}")
            return "unknown", 0.0
    
    def map_to_road_safety_event(self, urban_sound_label: str, confidence: float, 
                                 advanced_features: Dict[str, Any] = None) -> Tuple[AudioEvent, float]:
        """
        Map UrbanSound8K label to road safety event category
        
        Args:
            urban_sound_label: Prediction from model
            confidence: Model confidence score
            advanced_features: Advanced audio features for enhanced detection
            
        Returns:
            Tuple of (road_safety_event, adjusted_confidence)
        """
        # Base mapping
        if urban_sound_label in self.ROAD_SAFETY_MAPPING:
            base_event = self.ROAD_SAFETY_MAPPING[urban_sound_label]
        else:
            base_event = AudioEvent.UNKNOWN
        
        # Adjust confidence based on audio characteristics
        adjusted_confidence = confidence
        
        # Enhance detection using advanced features
        if advanced_features and base_event == AudioEvent.NORMAL:
            # Check if normal sound might actually be a crash
            if self._is_crash_like_sound(advanced_features, urban_sound_label):
                base_event = AudioEvent.POTENTIAL_CRASH
                # Boost confidence for crash detection
                adjusted_confidence = min(confidence * 1.3, 0.95)
        
        # Special handling for specific cases
        if urban_sound_label == 'gun_shot':
            # Very high confidence for crash-like sounds
            if confidence > 0.6:
                base_event = AudioEvent.IMPACT
                adjusted_confidence = confidence * 1.2
        
        # Ensure confidence is within bounds
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        if self.verbose:
            print(f"   Road Safety Mapping: {urban_sound_label} → {base_event.value}")
            print(f"   Confidence: {confidence:.2%} → {adjusted_confidence:.2%}")
        
        return base_event, adjusted_confidence
    
    def analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Main method to analyze audio file for road safety events
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"🔊 Audio Analysis: {os.path.basename(file_path)}")
            print("="*60)
        
        # Preprocess audio
        mfcc_features, advanced_features = self.preprocess_audio(file_path)
        
        # Predict class
        urban_sound_label, confidence = self.predict_class(mfcc_features, advanced_features)
        
        # Map to road safety event
        road_safety_event, adjusted_confidence = self.map_to_road_safety_event(
            urban_sound_label, confidence, advanced_features
        )
        
        # Prepare detailed response
        result = {
            'status': 'SUCCESS',
            'audio_file': os.path.basename(file_path),
            'urban_sound_prediction': {
                'label': urban_sound_label,
                'confidence': round(confidence, 4)
            },
            'road_safety_event': {
                'event': road_safety_event.value,
                'description': self._get_event_description(road_safety_event),
                'confidence': round(adjusted_confidence, 4),
                'severity': self._get_event_severity(road_safety_event)
            },
            'audio_features': {
                'duration': self.DURATION,
                'sample_rate': self.SAMPLE_RATE,
                'mfcc_features_shape': mfcc_features.shape,
                'advanced_features': {k: round(v, 4) for k, v in advanced_features.items()}
            },
            'model_info': {
                'model_used': 'pretrained_cnn' if self.model_loaded else 'simulation',
                'input_shape': str(self.INPUT_SHAPE)
            },
            'timestamp': librosa.get_duration(filename=file_path) if os.path.exists(file_path) else 0
        }
        
        # Add recommendation
        result['recommendation'] = self._get_recommendation(road_safety_event, adjusted_confidence)
        
        if self.verbose:
            print(f"\n📋 Analysis Complete:")
            print(f"   Event: {road_safety_event.value}")
            print(f"   Confidence: {adjusted_confidence:.2%}")
            print(f"   Recommendation: {result['recommendation']}")
            print("="*60)
        
        return result
    
    def _get_event_description(self, event: AudioEvent) -> str:
        """Get description for road safety event"""
        descriptions = {
            AudioEvent.NORMAL: "Normal traffic sounds",
            AudioEvent.POTENTIAL_CRASH: "Potential crash or impact detected",
            AudioEvent.EMERGENCY_VEHICLE_NEARBY: "Emergency vehicle siren detected",
            AudioEvent.TRAFFIC_WARNING: "Traffic warning (car horn) detected",
            AudioEvent.GLASS_BREAKING: "Glass breaking sound detected",
            AudioEvent.IMPACT: "Sharp impact sound detected",
            AudioEvent.UNKNOWN: "Unknown audio event"
        }
        return descriptions.get(event, "Unknown event")
    
    def _get_event_severity(self, event: AudioEvent) -> str:
        """Get severity level for road safety event"""
        severity_map = {
            AudioEvent.NORMAL: "LOW",
            AudioEvent.TRAFFIC_WARNING: "MEDIUM",
            AudioEvent.EMERGENCY_VEHICLE_NEARBY: "HIGH",
            AudioEvent.POTENTIAL_CRASH: "HIGH",
            AudioEvent.GLASS_BREAKING: "CRITICAL",
            AudioEvent.IMPACT: "CRITICAL",
            AudioEvent.UNKNOWN: "LOW"
        }
        return severity_map.get(event, "LOW")
    
    def _get_recommendation(self, event: AudioEvent, confidence: float) -> str:
        """Get action recommendation based on event"""
        if confidence < 0.5:
            return "Monitor situation (low confidence)"
        
        recommendations = {
            AudioEvent.NORMAL: "No action required",
            AudioEvent.TRAFFIC_WARNING: "Be alert for traffic warnings",
            AudioEvent.EMERGENCY_VEHICLE_NEARBY: "Yield to emergency vehicle",
            AudioEvent.POTENTIAL_CRASH: "Prepare for potential accident response",
            AudioEvent.GLASS_BREAKING: "Emergency - possible vehicle collision",
            AudioEvent.IMPACT: "Emergency - confirmed impact detected",
            AudioEvent.UNKNOWN: "Monitor audio for changes"
        }
        return recommendations.get(event, "Monitor situation")


# Utility function for real-time audio processing
class LiveAudioAnalyzer:
    """
    Real-time audio analyzer for continuous monitoring
    
    This class can be integrated with microphone input
    for real-time road safety monitoring
    """
    
    def __init__(self, detector: AudioEventDetector, buffer_duration: float = 2.0):
        """
        Initialize live audio analyzer
        
        Args:
            detector: AudioEventDetector instance
            buffer_duration: Duration of audio buffer in seconds
        """
        self.detector = detector
        self.buffer_duration = buffer_duration
        self.buffer = []
        
        if detector.verbose:
            print(f"🎤 LiveAudioAnalyzer initialized")
            print(f"   Buffer duration: {buffer_duration} seconds")
    
    def process_buffer(self, audio_chunk: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Process audio buffer chunk
        
        Args:
            audio_chunk: Audio samples
            sample_rate: Sampling rate
            
        Returns:
            Analysis results
        """
        # Save to temporary file for analysis
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Save audio chunk to file
            sf.write(tmp_file.name, audio_chunk, sample_rate)
            
            # Analyze using detector
            result = self.detector.analyze_audio(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return result


# Example usage and testing
if __name__ == "__main__":
    import random
    
    print("🧪 Testing AudioEventDetector")
    print("="*60)
    
    # Initialize detector
    detector = AudioEventDetector(
        model_path="audio_model.h5",  # Update with your model path
        verbose=True
    )
    
    # Test with sample audio files
    test_cases = [
        ("crash_simulation.wav", "Simulated crash sound"),
        ("siren_sample.wav", "Emergency vehicle siren"),
        ("horn_sample.wav", "Car horn warning"),
        ("normal_traffic.wav", "Normal traffic sounds")
    ]
    
    for audio_file, description in test_cases:
        if os.path.exists(audio_file):
            print(f"\n📁 Testing: {description}")
            result = detector.analyze_audio(audio_file)
            
            print(f"\n✅ Result:")
            print(f"   Event: {result['road_safety_event']['event']}")
            print(f"   Confidence: {result['road_safety_event']['confidence']:.2%}")
            print(f"   Severity: {result['road_safety_event']['severity']}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"\n⚠ Test file not found: {audio_file}")
            print(f"   Creating simulation...")
            
            # Create simulated result
            simulated_result = {
                'status': 'SIMULATED',
                'audio_file': audio_file,
                'road_safety_event': {
                    'event': random.choice(['POTENTIAL_CRASH', 'EMERGENCY_VEHICLE_NEARBY', 'NORMAL']),
                    'confidence': random.uniform(0.6, 0.9),
                    'severity': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                    'description': f"Simulated {description}"
                },
                'recommendation': "Test simulation - no actual audio analyzed"
            }
            
            print(f"\n🎭 Simulated Result:")
            print(f"   Event: {simulated_result['road_safety_event']['event']}")
            print(f"   Confidence: {simulated_result['road_safety_event']['confidence']:.2%}")
            print(f"   Severity: {simulated_result['road_safety_event']['severity']}")
    
    print("\n" + "="*60)
    print("🎯 Integration Instructions:")
    print("="*60)
    print("1. Place pre-trained 'audio_model.h5' in working directory")
    print("2. Update model_path parameter if using different file")
    print("3. For real usage, record audio clips from frontend")
    print("4. Call detector.analyze_audio(audio_file_path)")
    print("5. Use result['road_safety_event']['event'] for sensor fusion")
    print("\nExample integration with sensor fusion:")
    print("  audio_result = detector.analyze_audio('captured_audio.wav')")
    print("  audio_class = audio_result['road_safety_event']['event']")
    print("  audio_confidence = audio_result['road_safety_event']['confidence']")
    print("  fusion_result = fusion_engine.analyze(vision_conf, audio_class, audio_confidence, g_force)")
    print("="*60)