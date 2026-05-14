from collections import deque
import math
import json
import os
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
import pyttsx3
import platform


CONFIG_FILE = Path("config.json")

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
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


class GreetingDetector:
    def __init__(self, history_size: int = 8, x_movement_threshold: float = 0.05) -> None:
        self.hand_x_history = deque(maxlen=history_size)
        self.x_movement_threshold = x_movement_threshold
        self.last_trigger_time = 0.0
        self.cooldown_seconds = 3.0
        self.last_x = None
        self.last_direction = 0
        self.last_trigger_x = None

    def update(self, wrist_x: float, palm_open: bool) -> bool:
        if not palm_open:
            self.reset()
            return False

        self.hand_x_history.append(wrist_x)

        if self.last_x is None:
            self.last_x = wrist_x
            self.last_trigger_x = wrist_x
            return False

        delta = wrist_x - self.last_x
        if abs(delta) < 0.004:
            return False

        current_direction = 1 if delta > 0 else -1
        movement_from_last_trigger = abs(wrist_x - (self.last_trigger_x or wrist_x))

        should_trigger = (
            self.last_direction != 0
            and current_direction != self.last_direction
            and movement_from_last_trigger >= self.x_movement_threshold
        )

        self.last_x = wrist_x
        self.last_direction = current_direction

        now = time.time()
        if not should_trigger or now - self.last_trigger_time < self.cooldown_seconds:
            return False

        self.last_trigger_time = now
        self.last_x = None
        self.last_direction = 0
        self.last_trigger_x = None
        self.hand_x_history.clear()
        return True

    def reset(self) -> None:
        self.hand_x_history.clear()
        self.last_x = None
        self.last_direction = 0
        self.last_trigger_x = None
        self.last_trigger_time = 0.0


def is_open_palm(landmarks, handedness_label: str) -> bool:
    fingers = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    extended = 0
    for tip_index, pip_index in fingers:
        if landmarks[tip_index].y < landmarks[pip_index].y:
            extended += 1

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if handedness_label == "Right":
        thumb_extended = thumb_tip.x > thumb_ip.x
    else:
        thumb_extended = thumb_tip.x < thumb_ip.x

    if thumb_extended:
        extended += 1

    return extended >= 4


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


def ensure_model_file() -> Path:
    model_path = Path("models") / "hand_landmarker.task"
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    print("Descargando el modelo de manos por primera vez...")
    urllib.request.urlretrieve(model_url, model_path)
    return model_path


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

    def _check_piper_available(self) -> bool:
        """Check if piper executable is available by actually trying to run it."""
        piper_cmd = "piper.exe" if platform.system() == "Windows" else "piper"
        
        # First, try shutil.which()
        if shutil.which(piper_cmd) is not None:
            return True
        
        # On Linux, try common installation paths
        if platform.system() != "Windows":
            common_paths = [
                "/usr/local/bin/piper",
                "/usr/bin/piper",
                os.path.expanduser("~/.local/bin/piper"),
            ]
            for path in common_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    return True
        
        # Try to run piper --version as a fallback
        try:
            subprocess.run([piper_cmd, "--version"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         timeout=2)
            return True
        except Exception:
            pass
        
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
            if self._check_piper_available():
                self.piper_voice = True  # Flag: Piper está disponible
                print(f"TTSWorker: voz Piper cargada desde {self.piper_model_path}")
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
        piper_cmd = "piper.exe" if platform.system() == "Windows" else "piper"
        model_abs = os.path.abspath(str(self.piper_model_path)).replace(".onnx", "")
        
        piper_process = subprocess.Popen(
            [piper_cmd, "--model", model_abs, "--output-raw", "--length-scale", "0.7"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        audio_data, stderr_data = piper_process.communicate(input=text.encode())
        if not audio_data:
            raise RuntimeError(f"Piper no generó audio: {stderr_data.decode()}")
        
        # Guardar en caché
        try:
            with open(cache_path, "wb") as f:
                f.write(audio_data)
            print(f"TTSWorker: audio guardado en caché")
        except Exception as e:
            print(f"TTSWorker: no se pudo guardar en caché: {e}")
        
        return audio_data

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
                    with wave.open(str(temp_wav_path), "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(22050)
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
                    with wave.open(str(temp_wav_path), "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(22050)
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
            "capturing": "capturing",
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
        # Choose interval: speaking is fast, listening occasionally changes for life, others static
        if speaking:
            interval = self.speaking_interval
        elif mapped_state == "listening" and len(frames) > 1:
            interval = self.listening_interval  # longer interval for occasional blink
        else:
            interval = float('inf')  # never auto-advance if not speaking or listening multi-frame

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
    detectors = [GreetingDetector()]
    tts = TTSWorker(config)
    model_path = ensure_model_file()
    character = CharacterFaceLoader()
    show_camera_preview = bool(config.get("show_camera_preview", False))

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la camara del dispositivo.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    current_state = "listening"

    # Create and configure fullscreen window
    window_name = "Saludo con camara"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with vision.HandLandmarker.create_from_options(options) as hands:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("No se pudo leer la camara.")
                break

            frame = cv2.flip(frame, 1)
            detection_frame = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))
            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = hands.detect(mp_image)

            status_text = "Busca tu mano en la camara"
            status_color = (255, 255, 255)
            speech_triggered = False
            hands_detected = 0
            next_state = "listening"
            
            # Lógica de estados según actividad TTS
            if tts.is_synthesizing():
                # Fase de síntesis: mostrar "Procesando voz..." y estado capturing
                status_text = "Procesando voz..."
                status_color = (0, 200, 255)
                next_state = "capturing"
            elif tts.is_speaking():
                # Fase de reproducción: mostrar estado speaking (con animación de boca)
                status_text = "Hablando..."
                status_color = (0, 255, 100)
                next_state = "speaking"
            elif result.hand_landmarks and result.handedness:
                hands_detected = len(result.hand_landmarks)
                for hand_index in range(hands_detected):
                    hand_landmarks = result.hand_landmarks[hand_index]
                    handedness_label = result.handedness[hand_index][0].category_name
                    wrist_x = hand_center_x(hand_landmarks)
                    palm_open = is_open_palm(hand_landmarks, handedness_label)

                    if show_camera_preview:
                        for landmark in hand_landmarks:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    # Use the detector for this hand
                    hand_speech = detectors[hand_index].update(wrist_x, palm_open)
                    if hand_speech:
                        speech_triggered = True
                    
                    # Update status based on first hand for display
                    if hand_index == 0:
                        if hand_speech:
                            status_text = "Saludo"
                            status_color = (0, 255, 0)
                        elif palm_open:
                            status_text = "Mano detectada, mueve la mano para saludar"
                            status_color = (0, 200, 255)
                        else:
                            status_text = "Mano detectada"
                            status_color = (255, 200, 0)
                
                # Reset detectors for hands that are no longer detected
                for hand_index in range(hands_detected, len(detectors)):
                    detectors[hand_index].reset()
                
                if speech_triggered:
                    tts.say("Hola, Bienvenido al family day de Dia")
            else:
                for detector in detectors:
                    detector.reset()

            # Actualizar estado según actividad
            if next_state != current_state:
                current_state = next_state
                character.reset_state(current_state)

            speaking = tts.is_speaking()

            draw_label(frame, "Presiona q para salir", 30, (255, 255, 255))
            draw_label(frame, status_text, 70, status_color)

            character_img = character.next_frame(current_state, speaking=speaking)
            if character_img is not None:
                character_resized = resize_with_aspect_ratio(character_img, 1280, 720, (201, 227, 193))
                canvas = character_resized.copy()
                if show_camera_preview:
                    frame_resized = cv2.resize(frame, (320, 240))
                    cam_x = canvas.shape[1] - frame_resized.shape[1] - 10
                    cam_y = canvas.shape[0] - frame_resized.shape[0] - 10
                    canvas[cam_y:cam_y + frame_resized.shape[0], cam_x:cam_x + frame_resized.shape[1]] = frame_resized
            else:
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                placeholder[:] = [201, 227, 193]
                cv2.putText(placeholder, "Coloca PNGs en", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(placeholder, "faces/" + current_state, (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                canvas = placeholder.copy()
                if show_camera_preview:
                    frame_resized = cv2.resize(frame, (320, 240))
                    cam_x = canvas.shape[1] - frame_resized.shape[1] - 10
                    cam_y = canvas.shape[0] - frame_resized.shape[0] - 10
                    canvas[cam_y:cam_y + frame_resized.shape[0], cam_x:cam_x + frame_resized.shape[1]] = frame_resized

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
