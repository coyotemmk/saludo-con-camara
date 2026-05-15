import json
from pathlib import Path
import shutil
import urllib.request
import threading
import time
from queue import Queue
import subprocess
import tempfile
import wave
from typing import Optional
import hashlib

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from piper import PiperVoice, SynthesisConfig
import pyttsx3
import platform


CONFIG_FILE = Path("config.json")

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
DETECTION_WIDTH = 320
DETECTION_HEIGHT = 240

DEFAULT_CONFIG = {
    "tts_backend": "auto",
    "piper_voice_model": "voice/bmo.onnx",
    "piper_voice_config": "voice/bmo.onnx.json",
    "show_camera_preview": False,
    "system_voice_name": "",
    "system_voice_rate": 150,
    "system_voice_volume": 0.9,
    "system_language_hint": "es",
    "proximity_pose_area_threshold": 0.25,
    "proximity_cooldown_seconds": 5.0,
    "gesture_wrist_shoulder_margin": 0.03,
    "gesture_elbow_shoulder_margin": 0.03,
}


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
                user_config = json.load(config_file)
            if isinstance(user_config, dict):
                config.update(user_config)
        except Exception as exc:
            print(f"Config: no se pudo leer {CONFIG_FILE}: {exc}")
    return config


class PoseGestureDetector:
    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0.0
        self.is_active = False

    def update(self, gesture_active: bool) -> bool:
        now = time.time()

        if not gesture_active:
            self.is_active = False
            return False

        if self.is_active:
            return False

        if now - self.last_trigger_time < self.cooldown_seconds:
            return False

        self.is_active = True
        self.last_trigger_time = now
        return True

    def reset(self) -> None:
        self.is_active = False


def is_arm_raised(
    pose_landmarks,
    wrist_shoulder_margin: float = 0.03,
    elbow_shoulder_margin: float = 0.03,
) -> bool:
    """Detecta si uno de los brazos está levantado por encima del hombro."""
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    left_elbow = pose_landmarks[13]
    right_elbow = pose_landmarks[14]
    left_wrist = pose_landmarks[15]
    right_wrist = pose_landmarks[16]

    left_arm_up = left_wrist.y < left_shoulder.y - wrist_shoulder_margin and left_elbow.y < left_shoulder.y + elbow_shoulder_margin
    right_arm_up = right_wrist.y < right_shoulder.y - wrist_shoulder_margin and right_elbow.y < right_shoulder.y + elbow_shoulder_margin

    return left_arm_up or right_arm_up


def is_pose_too_close(pose_landmarks, pose_area_threshold: float = 0.25) -> bool:
    """Detecta si estás muy cerca usando solo el área del pose (cuerpo).
    
    Mide el bbox de los landmarks del pose y lo compara contra un umbral.
    Si el área del cuerpo > pose_area_threshold, significa que está muy cerca.
    """
    if not pose_landmarks:
        return False
    
    pxs = [lm.x for lm in pose_landmarks]
    pys = [lm.y for lm in pose_landmarks]
    p_xmin, p_xmax = min(pxs), max(pxs)
    p_ymin, p_ymax = min(pys), max(pys)
    pose_width = p_xmax - p_xmin
    pose_height = p_ymax - p_ymin
    pose_area = pose_width * pose_height
    
    return pose_area > pose_area_threshold


POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
)


def _normalized_to_pixel(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    x = int(max(0.0, min(1.0, float(landmark.x))) * frame_width)
    y = int(max(0.0, min(1.0, float(landmark.y))) * frame_height)
    return x, y


def draw_tracking_overlay(frame: np.ndarray, pose_result) -> np.ndarray:
    """Dibuja puntos y líneas de seguimiento sobre el frame de cámara."""
    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]

    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            for start_idx, end_idx in POSE_CONNECTIONS:
                start_landmark = pose_landmarks[start_idx]
                end_landmark = pose_landmarks[end_idx]
                start_point = _normalized_to_pixel(start_landmark, frame_width, frame_height)
                end_point = _normalized_to_pixel(end_landmark, frame_width, frame_height)
                cv2.line(annotated_frame, start_point, end_point, (255, 180, 0), 2)

            for landmark in pose_landmarks:
                x, y = _normalized_to_pixel(landmark, frame_width, frame_height)
                cv2.circle(annotated_frame, (x, y), 3, (0, 120, 255), -1)

    return annotated_frame


def hand_center_x(landmarks) -> float:
    anchor_points = (0, 5, 9, 13, 17)
    return sum(landmarks[index].x for index in anchor_points) / len(anchor_points)


def resize_with_aspect_ratio(image, target_width: int, target_height: int, background_color: tuple[int, int, int]) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)

    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized_image
    return canvas


def ensure_task_model_file(model_filename: str, model_url: str, description: str) -> Path:
    model_path = Path("models") / model_filename
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando el modelo de {description} por primera vez...")
    urllib.request.urlretrieve(model_url, model_path)
    return model_path


def ensure_pose_model_file() -> Path:
    return ensure_task_model_file(
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "pose",
    )


def find_piper_voice_model(config: dict) -> tuple[Optional[Path], Optional[Path]]:
    """Find piper voice model files, checking multiple locations."""
    model_path = Path(str(config.get("piper_voice_model", "voice/bmo.onnx")))
    config_path = Path(str(config.get("piper_voice_config", "voice/bmo.onnx.json")))

    # Try path as-is first (relative or absolute)
    if model_path.is_absolute():
        if model_path.exists() and config_path.exists():
            print(f"Piper: Modelo encontrado en {model_path}")
            return model_path, config_path
    else:
        # Relative path: resolve from current working directory
        resolved_model = Path.cwd() / model_path
        resolved_config = Path.cwd() / config_path
        if resolved_model.exists() and resolved_config.exists():
            print(f"Piper: Modelo encontrado en {resolved_model}")
            return resolved_model, resolved_config
    
    # Try alternative paths
    alternatives = [
        (Path("./voice/bmo.onnx"), Path("./voice/bmo.onnx.json")),
        (Path("./bmo.onnx"), Path("./bmo.onnx.json")),
        (Path("voice/es_ES-davefx-medium.onnx"), Path("voice/es_ES-davefx-medium.onnx.json")),
        (Path("./voice/es_ES-davefx-medium.onnx"), Path("./voice/es_ES-davefx-medium.onnx.json")),
    ]
    
    for alt_model, alt_config in alternatives:
        resolved_model = Path.cwd() / alt_model if not alt_model.is_absolute() else alt_model
        resolved_config = Path.cwd() / alt_config if not alt_config.is_absolute() else alt_config
        if resolved_model.exists() and resolved_config.exists():
            print(f"Piper: Modelo encontrado en {resolved_model}")
            return resolved_model, resolved_config
    
    print(f"Piper: No se encontró modelo")
    return None, None


class TTSWorker:
    def __init__(self, config: dict) -> None:
        self.queue: Queue = Queue()
        self.min_interval_seconds = 0.0
        self.last_spoken_at = 0.0
        self._speaking = threading.Event()
        self._synthesizing = threading.Event()  # Flag interno: síntesis en progreso
        self.engine = None
        self.piper_voice = None
        self.config = config
        self.tts_backend = str(config.get("tts_backend", "auto")).lower()
        self.piper_model_path, self.piper_config_path = find_piper_voice_model(config)
        
        # Configurar caché de audio
        self.cache_dir = Path("tts_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self._init_engine()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self._warm_cache_thread = threading.Thread(target=self._warm_cache, daemon=True)
        self._warm_cache_thread.start()

    def _load_piper_voice(self) -> bool:
        """Load Piper voice directly through the Python package to avoid piper.exe."""
        try:
            if self.piper_model_path is None:
                return False

            self.piper_voice = PiperVoice.load(
                self.piper_model_path,
                self.piper_config_path,
            )
            print(f"TTSWorker: voz Piper cargada desde {self.piper_model_path}")
            return True
        except Exception as exc:
            self.piper_voice = None
            print(f"TTSWorker: no se pudo cargar Piper desde Python - {exc}")
            return False

    def _init_engine(self) -> None:
        """Initialize pyttsx3 engine with Spanish voice if available."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', int(self.config.get("system_voice_rate", 150)))
            self.engine.setProperty('volume', float(self.config.get("system_voice_volume", 0.9)))
            self._select_spanish_voice()
            print("TTSWorker: pyttsx3 inicializado")
        except Exception as e:
            print(f"TTSWorker: Error inicializando motor TTS -> {e}")

        use_piper = self.tts_backend in ("auto", "piper")
        if use_piper and self.piper_model_path is not None:
            if self._load_piper_voice():
                pass
            else:
                self.piper_voice = None
                if self.tts_backend == "piper":
                    print("TTSWorker: no se pudo cargar Piper; usando pyttsx3")
                else:
                    print("TTSWorker: Piper no disponible; usando pyttsx3")
        else:
            if self.tts_backend == "piper":
                print("TTSWorker: modelo Piper no configurado; usando pyttsx3")

    def _select_spanish_voice(self) -> None:
        if not self.engine:
            return

        try:
            voices = self.engine.getProperty('voices')
        except Exception:
            return

        def voice_matches_spanish(voice) -> bool:
            voice_id = str(getattr(voice, 'id', '')).lower()
            voice_name = str(getattr(voice, 'name', '')).lower()
            languages = getattr(voice, 'languages', []) or []

            language_text = " ".join(
                str(language, errors='ignore').lower() if isinstance(language, (bytes, bytearray))
                else str(language).lower()
                for language in languages
            )

            return (
                'spanish' in voice_name
                or 'spanish' in voice_id
                or 'es_' in voice_id
                or 'es-' in voice_id
                or 'es' in language_text
                or 'spa' in language_text
            )

        selected_voice = None
        voice_name_hint = str(self.config.get("system_voice_name", "")).strip().lower()
        language_hint = str(self.config.get("system_language_hint", "es")).strip().lower()

        for voice in voices:
            if voice_matches_spanish(voice):
                if voice_name_hint:
                    if voice_name_hint in str(getattr(voice, 'name', '')).lower() or voice_name_hint in str(getattr(voice, 'id', '')).lower():
                        selected_voice = voice
                        break
                else:
                    selected_voice = voice
                    break

        if selected_voice is not None:
            self.engine.setProperty('voice', selected_voice.id)
            print(f"TTSWorker: voz española seleccionada -> {selected_voice.name}")
        elif voices:
            print("TTSWorker: no se encontró voz española, usando la primera voz disponible")
            self.engine.setProperty('voice', voices[0].id)

    def _run(self) -> None:
        print("TTSWorker: hilo iniciado")
        while True:
            text = self.queue.get()
            print(f"TTSWorker: procesando -> {text}")
            
            try:
                now = time.time()
                wait_seconds = self.min_interval_seconds - (now - self.last_spoken_at)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                if self.piper_voice:
                    try:
                        self._speak_with_piper(text)
                    except Exception as e:
                        print(f"TTSWorker: Piper falló, usando pyttsx3 - {e}")
                        if self.engine:
                            self._speaking.set()
                            self.engine.say(text)
                            self.engine.runAndWait()
                            self._speaking.clear()
                elif self.engine:
                    self._speaking.set()
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self._speaking.clear()
                self.last_spoken_at = time.time()
            except Exception as e:
                print(f"TTSWorker: error TTS -> {e}")
            finally:
                # Asegurar que se limpia el flag
                self._speaking.clear()
                try:
                    self.queue.task_done()
                except Exception:
                    pass

    def say(self, text: str) -> None:
        print(f"TTSWorker: encolando -> {text}")
        self.queue.put(text)

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def is_synthesizing(self) -> bool:
        """Retorna True si se está sintetizando audio (fase de texto a voz)."""
        return self._synthesizing.is_set()

    def _get_cache_path(self, text: str) -> Path:
        """Generar ruta de caché basada en hash MD5 del texto."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.wav"

    def _get_or_synthesize_audio(self, text: str) -> bytes:
        """Obtener audio del caché o sintetizar si no existe."""
        cache_path = self._get_cache_path(text)
        
        # Si existe en caché, cargar
        if cache_path.exists():
            print(f"TTSWorker: usando audio en caché para '{text[:40]}...'")
            with open(cache_path, "rb") as f:
                return f.read()
        
        # Si no existe, sintetizar
        print(f"TTSWorker: sintetizando nuevo audio para '{text[:40]}...'")
        if not self.piper_voice:
            raise RuntimeError("Piper no está disponible en memoria")

        syn_config = SynthesisConfig(length_scale=0.7)
        audio_parts = []
        for audio_chunk in self.piper_voice.synthesize(text, syn_config=syn_config):
            audio_parts.append(audio_chunk.audio_int16_bytes)

        audio_data = b"".join(audio_parts)
        if not audio_data:
            raise RuntimeError("Piper no generó audio")
        
        # Guardar en caché
        try:
            with open(cache_path, "wb") as f:
                f.write(audio_data)
            print(f"TTSWorker: audio guardado en caché")
        except Exception as e:
            print(f"TTSWorker: no se pudo guardar en caché: {e}")
        
        return audio_data

    def _warm_cache(self) -> None:
        """Pre-genera en segundo plano las frases más usadas con la misma voz Piper."""
        if not self.piper_voice:
            return

        phrases = [
            "Hola, Bienvenido al family day de Dia",
            "Jajaja, ¡qué guay! Tú sí que tienes estilo, parcero",
        ]

        for phrase in phrases:
            try:
                self._get_or_synthesize_audio(phrase)
            except Exception as exc:
                print(f"TTSWorker: no se pudo precargar caché para '{phrase[:30]}...': {exc}")

    def _speak_with_piper(self, text: str) -> None:
        """Synthesize and play audio using piper executable."""
        
        # Marcar fase de síntesis
        self._synthesizing.set()
        
        try:
            # Obtener audio del caché o sintetizar
            audio_data = self._get_or_synthesize_audio(text)
            
            # Síntesis completada, ahora reproducir
            self._synthesizing.clear()
            
            # Play audio on Windows
            if platform.system() == "Windows":
                import winsound
                temp_wav_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                        temp_wav_path = Path(temp_wav.name)
                    sample_rate = 22050
                    if self.piper_voice is not None:
                        sample_rate = int(getattr(self.piper_voice.config, "sample_rate", sample_rate))
                    with wave.open(str(temp_wav_path), "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_data)
                    # AQUÍ comienza reproducción real - activar flag "speaking"
                    self._speaking.set()
                    winsound.PlaySound(str(temp_wav_path), winsound.SND_FILENAME)
                finally:
                    # Limpiar después de reproducción
                    self._speaking.clear()
                    if temp_wav_path and temp_wav_path.exists():
                        try:
                            temp_wav_path.unlink()
                        except Exception:
                            pass
            else:
                # Play audio on Linux
                temp_wav_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                        temp_wav_path = Path(temp_wav.name)
                    sample_rate = 22050
                    if self.piper_voice is not None:
                        sample_rate = int(getattr(self.piper_voice.config, "sample_rate", sample_rate))
                    with wave.open(str(temp_wav_path), "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_data)
                    # AQUÍ comienza reproducción real - activar flag "speaking"
                    self._speaking.set()
                    if shutil.which("aplay") is not None:
                        subprocess.run(["aplay", "-q", str(temp_wav_path)], check=False)
                    elif shutil.which("paplay") is not None:
                        subprocess.run(["paplay", str(temp_wav_path)], check=False)
                    else:
                        raise RuntimeError("No se encontró aplay ni paplay")
                finally:
                    # Limpiar después de reproducción
                    self._speaking.clear()
                    if temp_wav_path and temp_wav_path.exists():
                        try:
                            temp_wav_path.unlink()
                        except Exception:
                            pass
        finally:
            # Asegurar que ambos flags se limpian en cualquier caso
            self._synthesizing.clear()
            self._speaking.clear()


def draw_label(frame, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


class CharacterFaceLoader:
    def __init__(self, faces_dir: str = "faces") -> None:
        self.faces_dir = Path(faces_dir)
        self.images = {}
        self.index = {}
        self.last_update = {}
        self.active_state = None
        # default frame intervals (seconds)
        self.frame_interval = 0.25
        self.speaking_interval = 0.15  # Más rápido para animación fluida de labios
        self.listening_interval = 2.0  # longer for occasional blink/life

        # mapping logical states from this app to folders
        self.state_mapping = {
            "idle": "idle",
            "greeting": "listening",
            "warmup": "warmup",
            "listening": "listening",
            "speaking": "speaking",
            "thinking": "thinking",
            "error": "error",
            # "capturing" expression removed; map to listening to avoid using that folder
            "capturing": "listening",
        }

        if self.faces_dir.exists():
            for state_dir in sorted(self.faces_dir.iterdir()):
                if state_dir.is_dir():
                    state_name = state_dir.name
                    pngs = sorted(state_dir.glob("*.png"))
                    loaded_imgs = []
                    for p in pngs:
                        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                        if img is None:
                            continue
                        # ensure 3-channel BGR
                        if img.shape[-1] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        loaded_imgs.append(img)
                    if loaded_imgs:
                        self.images[state_name] = loaded_imgs
                        self.index[state_name] = 0
                        self.last_update[state_name] = time.time()
                        # Info detallada para debugging
                        if state_name == "speaking":
                            print(f"✓ Cargados {len(loaded_imgs)} frames para '{state_name}': {[p.name for p in sorted(state_dir.glob('*.png'))]}")
                        else:
                            print(f"✓ Cargados {len(loaded_imgs)} frames para '{state_name}'")

    def get_frame(self, state: str):
        # kept for backward compatibility
        return self.next_frame(state, speaking=False)

    def reset_state(self, state: str) -> None:
        mapped_state = self.state_mapping.get(state, state)
        if mapped_state in self.index:
            self.index[mapped_state] = 0
            self.last_update[mapped_state] = time.time()
        self.active_state = mapped_state

    def next_frame(self, state: str, speaking: bool = False):
        mapped_state = self.state_mapping.get(state, state)
        frames = self.images.get(mapped_state)
        if not frames:
            return None

        if mapped_state != self.active_state:
            self.active_state = mapped_state
            if mapped_state not in self.index:
                self.index[mapped_state] = 0
            self.last_update[mapped_state] = 0.0

        now = time.time()
        # Choose interval: speaking is fast, thinking and listening occasionally changes for life, others static
        if speaking:
            interval = self.speaking_interval
        elif (mapped_state == "listening" or mapped_state == "thinking") and len(frames) > 1:
            interval = self.listening_interval  # longer interval for occasional blink/thinking animation
        else:
            interval = float('inf')  # never auto-advance if not speaking, listening or thinking

        last = self.last_update.get(mapped_state, 0.0)
        current_index = self.index.get(mapped_state, 0)

        if last == 0.0:
            self.last_update[mapped_state] = now
            return frames[current_index]

        # Advance frame if interval has passed
        if now - last >= interval:
            current_index = (current_index + 1) % len(frames)
            self.index[mapped_state] = current_index
            self.last_update[mapped_state] = now

        return frames[current_index]


def main() -> None:
    config = load_config()
    tts = TTSWorker(config)
    pose_model_path = ensure_pose_model_file()
    character = CharacterFaceLoader()
    show_camera_preview = bool(config.get("show_camera_preview", config.get("show_camara_preview", False)))

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la camara del dispositivo.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    current_state = "listening"
    face_presence_start = None  # timestamp cuando se detecta una sola persona por primera vez
    face_presence_threshold = 0.5  # segundos antes de cambiar a listening
    pose_gesture_detector = PoseGestureDetector(cooldown_seconds=3.0)
    # Saludo por detección de cara (cooldown para no repetir)
    last_face_greet_time = 0.0
    face_greet_cooldown = 10.0
    # Múltiples personas: mostrar capturing 1s antes de hablar
    multi_person_start_time = None
    multi_person_announced = False
    # Persona muy cerca: aviso de proximidad
    last_too_close_time = 0.0
    too_close_cooldown = float(config.get("proximity_cooldown_seconds", 5.0))
    proximity_pose_area_threshold = float(config.get("proximity_pose_area_threshold", 0.25))
    gesture_wrist_shoulder_margin = float(config.get("gesture_wrist_shoulder_margin", 0.03))
    gesture_elbow_shoulder_margin = float(config.get("gesture_elbow_shoulder_margin", 0.03))

    # Create and configure fullscreen window
    window_name = "Saludo con camara"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("No se pudo leer la camara.")
                break

            now = time.time()

            frame = cv2.flip(frame, 1)
            detection_frame = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_result = pose_landmarker.detect(mp_image)
            pose_count = len(pose_result.pose_landmarks) if pose_result.pose_landmarks is not None else 0

            status_text = "Busca tu rostro en la camara"
            status_color = (255, 255, 255)
            next_state = "thinking"
            
            # Lógica de estados según actividad TTS
            if tts.is_synthesizing():
                # Fase de síntesis: mostrar "Procesando voz..." (usar estado listening en lugar de capturing)
                status_text = "Procesando voz..."
                status_color = (0, 200, 255)
                next_state = "listening"
            elif tts.is_speaking():
                # Fase de reproducción: mostrar estado speaking (con animación de boca)
                status_text = "Hablando..."
                status_color = (0, 255, 100)
                next_state = "speaking"
            elif pose_count == 0:
                status_text = "No veo a nadie"
                status_color = (255, 200, 0)
                next_state = "thinking"
                face_presence_start = None
                pose_gesture_detector.reset()
                multi_person_start_time = None
                multi_person_announced = False
                last_too_close_time = 0.0
            elif pose_count > 1:
                # Múltiples personas detectadas: mostrar capturing 1s, luego hablar
                status_text = "Solo una persona, por favor"
                status_color = (100, 200, 255)
                next_state = "capturing"
                face_presence_start = None
                pose_gesture_detector.reset()
                last_too_close_time = 0.0

                # Primera vez que detectamos múltiples personas
                if multi_person_start_time is None:
                    multi_person_start_time = now
                    multi_person_announced = False

                # Después de 1 segundo, reproducir el aviso
                if not multi_person_announced and (now - multi_person_start_time) >= 1.0:
                    if not tts.is_synthesizing() and not tts.is_speaking():
                        tts.say("Hay muchas personas. Por favor, pónganse de uno en uno.")
                        multi_person_announced = True
            else:
                # Una sola persona: activar listening tras una pequeña estabilidad
                if face_presence_start is None:
                    # Si venimos de múltiples personas, permitir entrada inmediata a listening
                    if multi_person_start_time is not None:
                        # Ya detectamos múltiples antes, así que permitir entrada directa
                        face_presence_start = now - face_presence_threshold
                    else:
                        # Primera vez que vemos una persona
                        face_presence_start = time.time()

                time_with_face = time.time() - face_presence_start
                if time_with_face >= face_presence_threshold:
                    next_state = "listening"

                    # (No reproducir saludo automático aquí — el saludo se hará solo con el gesto)

                # Resetear múltiples personas cuando solo hay una
                multi_person_start_time = None
                multi_person_announced = False
                
                pose_is_visible = pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0
                
                # Detectar gesto de saludo PRIMERO
                arm_raised = False
                gesture_triggered = False
                if pose_is_visible:
                    pose_landmarks = pose_result.pose_landmarks[0]
                    arm_raised = is_arm_raised(
                        pose_landmarks,
                        gesture_wrist_shoulder_margin,
                        gesture_elbow_shoulder_margin,
                    )
                    if pose_gesture_detector.update(arm_raised):
                        status_text = "¡Hola! 👋"
                        status_color = (0, 255, 0)
                        next_state = "listening"
                        tts.say("Hola, Bienvenido al family day de Dia")
                        gesture_triggered = True
                    elif arm_raised:
                        status_text = "¡Hola! 👋"
                        status_color = (0, 255, 0)
                        next_state = "listening"
                
                # Advertencia de proximidad (solo usa pose)
                if pose_is_visible:
                    pose_lms = pose_result.pose_landmarks[0]
                    if is_pose_too_close(pose_lms, proximity_pose_area_threshold):
                        status_text = "Estás muy cerca"
                        status_color = (0, 120, 255)
                        if (now - last_too_close_time) >= too_close_cooldown:
                            if not tts.is_synthesizing() and not tts.is_speaking():
                                tts.say("Yepa, alejate un poco, estás muy cerca")
                                last_too_close_time = now
                    else:
                        status_text = "Te veo"
                        status_color = (0, 200, 255)

                # Resetear el detector si la persona se va o hay inestabilidad prolongada
                if time_with_face < face_presence_threshold:
                    next_state = "thinking"

            # Actualizar estado según actividad
            if next_state != current_state:
                current_state = next_state
                character.reset_state(current_state)

            speaking = tts.is_speaking()

            character_img = character.next_frame(current_state, speaking=speaking)
            if character_img is not None:
                character_resized = resize_with_aspect_ratio(character_img, 1280, 720, (201, 227, 193))
                canvas = character_resized.copy()
            else:
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                placeholder[:] = [201, 227, 193]
                cv2.putText(placeholder, "Coloca PNGs en", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(placeholder, "faces/" + current_state, (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                canvas = placeholder.copy()

            if show_camera_preview:
                frame_resized = cv2.resize(frame, (320, 240))
                preview_pose_result = pose_result
                frame_resized = draw_tracking_overlay(frame_resized, preview_pose_result)
                cam_x = canvas.shape[1] - frame_resized.shape[1] - 10
                cam_y = canvas.shape[0] - frame_resized.shape[0] - 10
                canvas[cam_y:cam_y + frame_resized.shape[0], cam_x:cam_x + frame_resized.shape[1]] = frame_resized

            draw_label(canvas, "Presiona q para salir", 30, (255, 255, 255))
            draw_label(canvas, status_text, 70, status_color)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
