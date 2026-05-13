from collections import deque
import math
from pathlib import Path
import urllib.request
import threading
import time
from queue import Queue

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3


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


class TTSWorker:
    def __init__(self) -> None:
        self.queue: Queue = Queue()
        self.min_interval_seconds = 0.0
        self.last_spoken_at = 0.0
        self._speaking = threading.Event()
        self.engine = None
        self._init_engine()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _init_engine(self) -> None:
        """Initialize pyttsx3 engine with Spanish voice if available."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            print("TTSWorker: pyttsx3 inicializado")
        except Exception as e:
            print(f"TTSWorker: Error inicializando motor TTS -> {e}")

    def _run(self) -> None:
        print("TTSWorker: hilo iniciado (pyttsx3)")
        while True:
            text = self.queue.get()
            print(f"TTSWorker: procesando -> {text}")
            try:
                now = time.time()
                wait_seconds = self.min_interval_seconds - (now - self.last_spoken_at)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                self._speaking.set()
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                self.last_spoken_at = time.time()
            except Exception as e:
                print(f"TTSWorker: error TTS -> {e}")
            finally:
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
        self.speaking_interval = 0.15
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
    detectors = [GreetingDetector(), GreetingDetector()]  # one detector per hand
    tts = TTSWorker()
    model_path = ensure_model_file()
    character = CharacterFaceLoader()

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la camara del dispositivo.")

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = hands.detect(mp_image)

            status_text = "Busca tu mano en la camara"
            status_color = (255, 255, 255)
            speech_triggered = False
            hands_detected = 0

            if result.hand_landmarks and result.handedness:
                hands_detected = len(result.hand_landmarks)
                for hand_index in range(hands_detected):
                    hand_landmarks = result.hand_landmarks[hand_index]
                    handedness_label = result.handedness[hand_index][0].category_name
                    wrist_x = hand_center_x(hand_landmarks)
                    palm_open = is_open_palm(hand_landmarks, handedness_label)

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
                    tts.say("Hola, Bienvenido al family day de Día")
            else:
                for detector in detectors:
                    detector.reset()
                status_text = "Busca tu mano en la camara"
                status_color = (255, 255, 255)

            speaking = tts.is_speaking()
            next_state = "speaking" if speaking else "listening"
            if next_state != current_state:
                current_state = next_state
                character.reset_state(current_state)

            if speaking and not speech_triggered:
                status_text = "Hablando"
                status_color = (0, 255, 0)

            draw_label(frame, "Presiona q para salir", 30, (255, 255, 255))
            draw_label(frame, status_text, 70, status_color)

            character_img = character.next_frame(current_state, speaking=speaking)
            if character_img is not None:
                character_resized = resize_with_aspect_ratio(character_img, 1280, 720, (201, 227, 193))
                frame_resized = cv2.resize(frame, (320, 240))
                
                # Overlay camera in bottom-right corner
                canvas = character_resized.copy()
                cam_x = canvas.shape[1] - frame_resized.shape[1] - 10
                cam_y = canvas.shape[0] - frame_resized.shape[0] - 10
                canvas[cam_y:cam_y + frame_resized.shape[0], cam_x:cam_x + frame_resized.shape[1]] = frame_resized
            else:
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                placeholder[:] = [201, 227, 193]
                cv2.putText(placeholder, "Coloca PNGs en", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.putText(placeholder, "faces/" + current_state, (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                
                frame_resized = cv2.resize(frame, (320, 240))
                canvas = placeholder.copy()
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
