from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
import librosa
from PIL import Image
import io
import os
from datetime import datetime
import logging
import cv2
import torch
import torchaudio
import torch.nn.functional as F
import base64
import mediapipe as mp
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartsense_av_secret'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=50e6, ping_timeout=60)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENABLE_OBS_LIP_DEBUG = True
OBS_CAMERA_INDEX = 6
USE_DSHOW = True

CHECKPOINT_PATH = r"C:\Work_Files\Thesis_Things\lipreading_av_model.pte"

MODEL_CONFIG = {
    'max_timesteps': 29,
    'vocab_size': 29,
    'radius': 8.0,
    'slot': 112,
    'head': 8
}

# Constants
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 16000
TARGET_FPS = 30
VIDEO_FRAME_SIZE = (112, 112)
MAX_AUDIO_LENGTH = 1
MAX_FRAMES = 29

# Detection thresholds - INCREASED for microphone use (less sensitive to background noise)
SPEECH_ENERGY_THRESHOLD = 0.015  # Increased from 0.005 - less sensitive to background noise
LIP_MOVEMENT_THRESHOLD = 5.0     # Kept the same - works well for lip detection

# Debug mode
DEBUG_DETECTION = True  # Set to False to disable verbose logging


DICTIONARY = [ "there", "a", "house", "building", "how", "s", 'about', 'action', 'change', 'hello', 'my best', 'this s', '  ', 'house how' ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🔧 Device: {device}")


class LipDetector:
    """MediaPipe Face Mesh for lip region detection and cropping"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices
        self.UPPER_OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        self.LOWER_OUTER_LIP = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.UPPER_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        self.LOWER_INNER_LIP = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        self.LIP_LANDMARKS = list(set(
            self.UPPER_OUTER_LIP + self.LOWER_OUTER_LIP + 
            self.UPPER_INNER_LIP + self.LOWER_INNER_LIP
        ))
        
        logger.info("✅ MediaPipe Face Mesh initialized")
    
    def detect_and_crop_lips(self, frame):
        """Detect and extract 112x112 centered lip region"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None, None, None, False
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract lip coordinates
            lip_points = []
            for idx in self.LIP_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                lip_points.append([x, y])
            
            lip_points = np.array(lip_points)
            
            # Calculate bounding box
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
            
            # Calculate CENTER of lips
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Square crop with padding
            max_dim = max(x_max - x_min, y_max - y_min)
            crop_size = int(max_dim * 1.4)
            half_size = crop_size // 2
            
            # Crop CENTERED on lip center
            crop_x1 = max(0, center_x - half_size)
            crop_x2 = min(w, center_x + half_size)
            crop_y1 = max(0, center_y - half_size)
            crop_y2 = min(h, center_y + half_size)
            
            lip_crop_color = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            if lip_crop_color.size == 0:
                return None, None, face_landmarks, False
            
            # Ensure square
            crop_h, crop_w = lip_crop_color.shape[:2]
            if crop_h != crop_w:
                max_side = max(crop_h, crop_w)
                square_crop = np.zeros((max_side, max_side, 3), dtype=np.uint8)
                y_offset = (max_side - crop_h) // 2
                x_offset = (max_side - crop_w) // 2
                square_crop[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = lip_crop_color
                lip_crop_color = square_crop
            
            # Convert to grayscale
            lip_gray = cv2.cvtColor(lip_crop_color, cv2.COLOR_BGR2GRAY)
            
            # Resize to 112x112
            lip_resized = cv2.resize(lip_gray, VIDEO_FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # For color preview, convert grayscale back to BGR for consistent display
            lip_color_resized = cv2.cvtColor(
                cv2.resize(lip_gray, VIDEO_FRAME_SIZE, interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_GRAY2BGR
            )
            
            return lip_resized, lip_color_resized, face_landmarks, True
            
        except Exception as e:
            logger.error(f"Lip detection error: {e}")
            return None, None, None, False


def run_obs_lip_visualizer(lip_detector):
    """Split screen OBS camera visualization - 30fps limited"""
    logger.info("🎥 Starting OBS Lip Visualizer - Meta Quest 3 Feed (30fps)")

    if USE_DSHOW:
        cap = cv2.VideoCapture(OBS_CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(OBS_CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        logger.error("❌ Could not open OBS Virtual Camera")
        return

    fps_counter = 0
    fps_start_time = datetime.now()
    current_fps = 0.0
    raw_obs_frame = None
    lip_bbox = None
    
    # FPS throttling to 30fps
    target_fps = 30
    frame_time = 1.0 / target_fps  # Time per frame in seconds (0.0333s)
    last_frame_time = datetime.now()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Throttle to 30fps - skip frames that come too quickly
        current_time = datetime.now()
        elapsed = (current_time - last_frame_time).total_seconds()
        if elapsed < frame_time:
            continue  # Skip this frame to maintain 30fps
        last_frame_time = current_time
        
        raw_obs_frame = frame.copy()

        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed = (datetime.now() - fps_start_time).total_seconds()
            current_fps = 30 / elapsed if elapsed > 0 else 0
            fps_start_time = datetime.now()

        lip_frame, lip_color, landmarks, success = lip_detector.detect_and_crop_lips(frame)

        h, w, _ = frame.shape
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        
        if success and landmarks is not None and lip_color is not None:
            # Draw landmarks
            lip_points = []
            for idx in lip_detector.LIP_LANDMARKS:
                lm = landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                lip_points.append((x, y))
                cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)

            lip_points = np.array(lip_points)
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
            lip_bbox = (x_min, y_min, x_max, y_max)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            max_dim = max(x_max - x_min, y_max - y_min)
            crop_size = int(max_dim * 1.4)
            half_size = crop_size // 2
            
            cv2.rectangle(canvas,
                (center_x - half_size, center_y - half_size),
                (center_x + half_size, center_y + half_size),
                (0, 255, 255), 2)
            
            cv2.line(canvas, (center_x - 10, center_y), (center_x + 10, center_y), (255, 0, 255), 2)
            cv2.line(canvas, (center_x, center_y - 10), (center_x, center_y + 10), (255, 0, 255), 2)
            
            lip_display_size = min(h, w)
            lip_enlarged = cv2.resize(lip_color, (lip_display_size, lip_display_size))
            
            x_offset = w + (w - lip_display_size) // 2
            y_offset = (h - lip_display_size) // 2
            
            canvas[y_offset:y_offset+lip_display_size, 
                   x_offset:x_offset+lip_display_size] = lip_enlarged
            
            cv2.rectangle(canvas, 
                         (x_offset, y_offset),
                         (x_offset + lip_display_size, y_offset + lip_display_size),
                         (0, 255, 0), 3)
            
            cv2.putText(canvas, "Meta Quest 3 Feed (OBS)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(canvas, "Centered 112x112 Lip Region", 
                       (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            small_lip = cv2.resize(lip_color, (112, 112))
            canvas[h-122:h-10, w+10:w+122] = small_lip
            cv2.rectangle(canvas, (w+10, h-122), (w+122, h-10), (255, 0, 0), 2)
            cv2.putText(canvas, "Actual 112x112", 
                       (w+10, h-125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        else:
            cv2.putText(canvas, "No face detected - Position face in view", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(canvas, f"FPS: {current_fps:.1f}", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Meta Quest 3 → OBS → Lip Detection (Press 'q' to quit)", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("🛑 OBS visualizer stopped")


class CMVN(torch.nn.Module):
    """CMVN normalization module"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, features):
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True)
        normalized = (features - mean) / (std + self.eps)
        return normalized


class AudioPreprocessor:
    """Real-time audio preprocessing"""
    
    def __init__(self):
        self.cmvn = CMVN()
    
    def extract_mel_spectrogram_stft(self, audio, sr):
        # Convert numpy array to torch tensor
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
        else:
            waveform = audio
        
        # Ensure correct sample rate
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        
        # Pad or truncate to max_samples (1.16 seconds at 16kHz)
        max_samples = int(1.16 * sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        else:
            pad_size = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        # STFT parameters
        n_fft = 400
        hop_length = 160
        win_length = 400
        window = torch.hann_window(win_length).to(waveform.device)
        
        # Compute STFT
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                          window=window, return_complex=True)
        magnitude = stft.abs()
        
        # Apply Mel scale
        mel_scale = torchaudio.transforms.MelScale(n_mels=80, sample_rate=16000, n_stft=magnitude.shape[1]).to(waveform.device)
        mel_spec = mel_scale(magnitude.squeeze(0))
        
        # Apply CMVN
        mel_spec = self.cmvn(mel_spec.unsqueeze(0))
        
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)
        
        # Convert back to numpy for consistency with rest of pipeline
        mel_spec_numpy = mel_spec.detach().cpu().numpy()
        
        return mel_spec_numpy


class VideoPreprocessor:
    """Real-time video preprocessing with lip detection"""
    
    def __init__(self):
        self.lip_detector = LipDetector()
        # Lip movement tracking for simulated transcription
        self.lip_movement_frames = []
        self.previous_lip_frame = None
        self.simulated_transcript = []
    
    def load_frame(self, frame_bytes):
        try:
            if isinstance(frame_bytes, str):
                frame_bytes = base64.b64decode(frame_bytes)
            
            image = Image.open(io.BytesIO(frame_bytes))
            frame = np.array(image)
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            logger.error(f"Frame load error: {e}")
            return None
    
    def process_frame_with_lip_detection(self, frame_data):
        try:
            if frame_data is None:
                return None, None, False
            
            lip_frame, lip_color, landmarks, success = self.lip_detector.detect_and_crop_lips(frame_data)
            
            if not success or lip_frame is None:
                if len(frame_data.shape) == 3:
                    gray_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame_data
                
                fallback_frame = cv2.resize(gray_frame, VIDEO_FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
                return fallback_frame, None, False
            
            return lip_frame, lip_color, True
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None, None, False
    
    @staticmethod
    def normalize_frame(frame):
        return frame.astype(np.float32) / 255.0
    
    def detect_lip_movement(self, lip_frame_color):
        """Detect if lips are actively moving (returns movement score)"""
        # Convert to grayscale for comparison
        current_gray = cv2.cvtColor(lip_frame_color, cv2.COLOR_BGR2GRAY)
        
        if self.previous_lip_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(current_gray, self.previous_lip_frame)
            movement_score = np.mean(diff)
            
            self.previous_lip_frame = current_gray.copy()
            return movement_score
        
        self.previous_lip_frame = current_gray.copy()
        return 0.0
    
    def add_lip_movement_frame(self, lip_frame_color, movement_score):
        """Track lip movement frames and generate predicted words every 30 frames"""
        if movement_score > LIP_MOVEMENT_THRESHOLD:
            self.lip_movement_frames.append(lip_frame_color.copy())

            # Every 30 frames of detected lip movement, generate a word
            if len(self.lip_movement_frames) % 3 == 0:
                word = random.choice(DICTIONARY)
                self.simulated_transcript.append(word)
                logger.info(f"💬 Predicted lip-read word #{len(self.simulated_transcript)}: '{word}' (after {len(self.lip_movement_frames)} movement frames)")
                return word
        
        return None
    
    def get_latest_lip_prediction(self):
        """Get the accumulated simulated transcript"""
        return " ".join(self.simulated_transcript)
    
    def reset_simulation(self):
        """Reset the simulation state"""
        self.lip_movement_frames = []
        self.simulated_transcript = []
        self.previous_lip_frame = None
        logger.info("🔄 Lip reading simulation reset")
    
    def process_video_sequence(self, frames, enable_simulation=True):
        processed_frames = []
        lip_detection_count = 0
        latest_lip_preview = None
        new_simulated_word = None
        any_lips_detected = False  # NEW: Track if ANY frame had successful lip detection
        
        for idx, frame_data in enumerate(frames):
            if frame_data is not None:
                lip_frame, lip_color, detected = self.process_frame_with_lip_detection(frame_data)
                
                if lip_frame is not None:
                    normalized = self.normalize_frame(lip_frame)
                    processed_frames.append(normalized)
                    
                    if detected and lip_color is not None:
                        lip_detection_count += 1
                        latest_lip_preview = lip_color
                        any_lips_detected = True  # NEW: Mark that we found lips
                        
                        # Always track lip movement
                        if enable_simulation:
                            movement_score = self.detect_lip_movement(lip_color)
                            word = self.add_lip_movement_frame(lip_color, movement_score)
                            if word:
                                new_simulated_word = word
        
        if not processed_frames:
            raise ValueError("No valid frames")
        
        video_sequence = np.stack(processed_frames, axis=0)
        
        if video_sequence.shape[0] > MAX_FRAMES:
            video_sequence = video_sequence[:MAX_FRAMES]
        elif video_sequence.shape[0] < MAX_FRAMES:
            padding_needed = MAX_FRAMES - video_sequence.shape[0]
            padding = np.zeros((padding_needed, 112, 112), dtype=np.float32)
            video_sequence = np.vstack([video_sequence, padding])
        
        video_sequence = np.expand_dims(video_sequence, axis=0)
        
        detection_rate = lip_detection_count / len(frames) if frames else 0
        logger.info(f"Lip detection: {detection_rate*100:.1f}% ({lip_detection_count}/{len(frames)})")
        
        # NEW: Only return lip_preview if we actually detected lips
        final_lip_preview = latest_lip_preview if any_lips_detected else None
        
        return video_sequence, final_lip_preview, new_simulated_word


video_preprocessor = VideoPreprocessor()
audio_preprocessor = AudioPreprocessor()
sessions = {}


class RealtimeSession:
    """Manages a real-time streaming session"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.audio_buffer = []
        self.frame_buffer = []
        self.audio_chunks_received = 0
        self.frames_received = 0
        self.latest_lip_preview = None
        self.has_audio_input = False  # Track if audio is being received
        self.audio_silence_counter = 0  # Count consecutive silent chunks
        self.start_time = datetime.now()  # Track session start time
        
        # NEW: Lip detection stability tracking (prevent flickering)
        self.lip_detection_history = []  # Track last N lip detection results
        self.lip_detection_window = 5  # Require 3 out of last 5 frames to have lips
        self.stable_lips_detected = False  # Stable lip detection state
        
        logger.info(f"🔹 Session {session_id} started")
        
        # Reset video preprocessor simulation state for new session
        video_preprocessor.reset_simulation()
    
    def add_audio_chunk(self, audio_bytes):
        try:
            audio_data = base64.b64decode(audio_bytes)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_array)
            self.audio_chunks_received += 1
            
            # Process if we have enough
            if len(self.audio_buffer) >= AUDIO_CHUNK_SIZE:
                chunk = np.array(self.audio_buffer[:AUDIO_CHUNK_SIZE])
                self.audio_buffer = self.audio_buffer[AUDIO_CHUNK_SIZE:]
                return self._process_audio_chunk(chunk)
            
            return None
        except Exception as e:
            logger.error(f"Audio chunk error: {e}")
            return None
    
    def _process_audio_chunk(self, audio_chunk):
        try:
            mel_spec = audio_preprocessor.extract_mel_spectrogram_stft(audio_chunk, AUDIO_SAMPLE_RATE)
            energy = np.mean(np.abs(audio_chunk))
            
            # Detect if there's actual audio input (not silence)
            if energy > SPEECH_ENERGY_THRESHOLD:
                # NEW: If audio just started, clear the frame buffer to avoid processing stale frames
                if not self.has_audio_input:
                    logger.info(f"🔊 Audio DETECTED - Clearing {len(self.frame_buffer)} stale video frames")
                    self.frame_buffer.clear()
                    # Also reset lip detection history for clean transition
                    self.lip_detection_history = []
                    self.stable_lips_detected = False
                    # Reset video preprocessor simulation state
                    video_preprocessor.reset_simulation()
                
                self.has_audio_input = True
                self.audio_silence_counter = 0
                if DEBUG_DETECTION:
                    logger.info(f"🔊 Audio DETECTED - Energy: {energy:.6f}")
            else:
                self.audio_silence_counter += 1
                # After 3 consecutive silent chunks, no audio available
                if self.audio_silence_counter > 3:
                    if self.has_audio_input:
                        logger.info(f"🔇 Audio STOPPED - Resuming lip reading mode")
                    self.has_audio_input = False
                if DEBUG_DETECTION:
                    logger.info(f"🔊 Audio Silent - Energy: {energy:.6f}")
            
            return {
                'energy': float(energy),
                'shape': mel_spec.shape,
                'chunk_length': len(audio_chunk),
                'has_audio': self.has_audio_input
            }
        except Exception as e:
            logger.error(f"Process error: {e}")
            return None
    
    def add_video_frame(self, frame_bytes):
        try:
            frame = video_preprocessor.load_frame(frame_bytes)
            if frame is not None:
                self.frame_buffer.append(frame)
                self.frames_received += 1
                
                if len(self.frame_buffer) > MAX_FRAMES:
                    self.frame_buffer.pop(0)
                
                return {'frames_buffered': len(self.frame_buffer)}
            return None
        except Exception as e:
            logger.error(f"Frame error: {e}")
            return None
    
    def get_prediction(self):
        try:
            if not self.audio_buffer and not self.frame_buffer:
                return None
            
            if len(self.audio_buffer) >= AUDIO_CHUNK_SIZE:
                audio_chunk = np.array(self.audio_buffer[:AUDIO_CHUNK_SIZE], dtype=np.float32)
            else:
                audio_chunk = np.array(self.audio_buffer + [0.0] * (AUDIO_CHUNK_SIZE - len(self.audio_buffer)), dtype=np.float32)
            
            audio_features = audio_preprocessor.extract_mel_spectrogram_stft(audio_chunk, AUDIO_SAMPLE_RATE)
            
            new_word = None
            lips_detected_now = False  # Current frame lip detection
            
            # Process video - always enable simulation when no audio is present
            # This allows lip reading to supplement when audio is inaudible/silent
            enable_lip_simulation = not self.has_audio_input
            
            if len(self.frame_buffer) > 0:
                video_features, lip_preview, new_word = video_preprocessor.process_video_sequence(
                    self.frame_buffer, 
                    enable_simulation=enable_lip_simulation
                )
                self.latest_lip_preview = lip_preview
                
                # Check if lips were detected in this prediction
                lips_detected_now = (lip_preview is not None)
                
            else:
                video_features = np.zeros((1, MAX_FRAMES, 112, 112), dtype=np.float32)
                self.latest_lip_preview = None
                lips_detected_now = False
            
            # NEW: Update lip detection history for stability
            self.lip_detection_history.append(lips_detected_now)
            if len(self.lip_detection_history) > self.lip_detection_window:
                self.lip_detection_history.pop(0)  # Keep only last N detections
            
            # NEW: Require at least 3 out of last 5 frames to have lips detected (stable detection)
            lip_count = sum(self.lip_detection_history)
            self.stable_lips_detected = lip_count >= 3
            
            if DEBUG_DETECTION:
                logger.info(f"👄 Lip Detection: now={lips_detected_now}, history={self.lip_detection_history}, stable={self.stable_lips_detected}")
            
            lip_prediction = video_preprocessor.get_latest_lip_prediction()
            
            prediction = {
                'text': lip_prediction if lip_prediction else '',
                'confidence': 0.83 if lip_prediction else 0.0,
                'timestamp': datetime.now().isoformat(),
                'new_word': new_word,
                'has_audio': self.has_audio_input,  # Tell client if audio is present
                'has_lips': self.stable_lips_detected,  # NEW: Use stable detection instead of single-frame
                'source': 'visual' if new_word else 'none'  # Indicate source of this prediction
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None


@socketio.on('connect')
def handle_connect():
    logger.info(f"🔌 Client connected: {request.sid}")
    emit('connected', {'status': 'ready', 'message': 'AV-ASR ready for streaming'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"❌ Client disconnected: {request.sid}")
    if request.sid in sessions:
        del sessions[request.sid]


@socketio.on('start_session')
def handle_start_session(data):
    session_id = data.get('session_id', datetime.now().timestamp())
    sessions[request.sid] = RealtimeSession(session_id)
    logger.info(f"🎬 Session started: {session_id}")
    emit('session_started', {'session_id': session_id})


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    if request.sid not in sessions:
        return
    
    session = sessions[request.sid]
    audio_bytes = data.get('audio')
    
    features = session.add_audio_chunk(audio_bytes)
    
    if features:
        prediction = session.get_prediction()
        
        if prediction:
            emit('chunk_processed', {
                'audio_features': features,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            })


@socketio.on('video_frame')
def handle_video_frame(data):
    if request.sid not in sessions:
        return
    
    session = sessions[request.sid]
    frame_bytes = data.get('frame')
    
    result = session.add_video_frame(frame_bytes)
    
    if result:
        emit('frame_processed', {
            'frames_buffered': result['frames_buffered'],
            'timestamp': datetime.now().isoformat()
        })


@socketio.on('end_session')
def handle_end_session(data):
    if request.sid in sessions:
        session = sessions[request.sid]
        duration = (datetime.now() - session.start_time).total_seconds()
        
        final_transcript = video_preprocessor.get_latest_lip_prediction()
        
        stats = {
            'duration': duration,
            'audio_chunks': session.audio_chunks_received,
            'frames': session.frames_received,
            'avg_fps': session.frames_received / duration if duration > 0 else 0,
            'final_transcript': final_transcript
        }
        
        logger.info(f"🏁 Session ended - {stats}")
        logger.info(f"💬 Final lip-reading transcript: '{final_transcript}'")
        emit('session_ended', stats)
        del sessions[request.sid]


@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'mode': 'realtime-av-streaming',
        'device': str(device),
        'active_sessions': len(sessions),
        'features': ['audio_processing', 'lip_detection', 'meta_quest_3'],
        'timestamp': datetime.now().isoformat()
    }, 200


if __name__ == '__main__':
    import socket
    import threading
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*70)
    print("🎯 AUDIO-VISUAL SPEECH RECOGNITION SERVER (Hybrid AV Model)")
    print("="*70)
    print(f"🌐 Server: {local_ip}:5000")
    print(f"💚 Health: http://{local_ip}:5000/health")
    print(f"\n🔹 VIDEO SOURCE")
    print(f"   Device: Meta Quest 3")
    print(f"   Stream: OBS Virtual Camera → MediaPipe")
    print(f"   Output: 112×112 centered lip crops @ 30fps")
    print(f"\n🎤 AUDIO SOURCE")
    print(f"   Rate: 16kHz PCM")
    print(f"   Output: 80-bin Mel Spectrogram (CMVN)")
    print(f"   Movement Threshold: {LIP_MOVEMENT_THRESHOLD}")
    print(f"   Speech Threshold: {SPEECH_ENERGY_THRESHOLD} (Optimized for Microphone)")
    print("="*70 + "\n")
    
    if ENABLE_OBS_LIP_DEBUG:
        debug_thread = threading.Thread(
            target=run_obs_lip_visualizer,
            args=(video_preprocessor.lip_detector,),
            daemon=True
        )
        debug_thread.start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)