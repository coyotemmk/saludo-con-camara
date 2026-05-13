# 🤖 Saludo con Cámara - Guía de Instalación

## 📋 Requisitos

- **Python 3.8+**
- **Cámara conectada** al dispositivo
- **Micrófono y altavoz** (para audio)

### Sistema Operativo
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, Debian, Raspberry Pi OS)
- ✅ macOS (parcial)

---

## 🚀 Instalación Rápida

### En Windows

1. **Descarga el proyecto** y abre PowerShell en la carpeta
2. **Ejecuta el instalador:**
   ```powershell
   .\setup.bat
   ```
3. **Inicia la app:**
   ```powershell
   .\start.bat
   ```
   O simplemente haz doble click en `start.bat`

### En Linux / Raspberry Pi

1. **Descarga el proyecto** y abre una terminal en la carpeta
2. **Dale permisos de ejecución:**
   ```bash
   chmod +x setup.sh start.sh
   ```
3. **Ejecuta el instalador:**
   ```bash
   ./setup.sh
   ```
4. **Inicia la app:**
   ```bash
   ./start.sh
   ```

---

## 📦 Instalación Manual

Si prefieres configurar todo manualmente:

```bash
# Crear entorno virtual
python -m venv venv

# Activar
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app.py
```

---

## 🎨 Estructura de Carpetas

```
Saludo con camara/
├── app.py                 # Aplicación principal
├── faces/                 # Animaciones del personaje
│   ├── listening/         # listen 01.png, listen 02.png
│   ├── speaking/          # speaking 01.png, speaking 02.png, speaking 03.png
│   ├── idle/
│   ├── thinking/
│   └── ...
├── models/                # Modelos de IA (se descargan automáticamente)
│   └── hand_landmarker.task
├── requirements.txt       # Dependencias Python
├── setup.sh / setup.bat   # Instaladores
└── start.sh / start.bat   # Scripts de ejecución
```

---

## ⚙️ Configuración

### Ajustar parámetros (en `app.py`):

```python
# Detección de gestos
cooldown_seconds = 3.0              # Espera entre saludos (en línea 23)
x_movement_threshold = 0.05         # Sensibilidad del vaivén

# Animación
listening_interval = 2.0            # Cambio de cara cada 2 seg (línea 207)
speaking_interval = 0.15            # Animación rápida al hablar
```

### Cambiar el texto de saludo:

Busca en `app.py` la línea con `tts.say()` y modifica:
```python
tts.say("Hola, Bienvenido al family day de Día")  # Cambia este texto
```

---

## 🔊 Idiomas y Voces

la app usa **pyttsx3** que soporta múltiples idiomas según tu SO:

- **Windows:** Voces instaladas del sistema
- **Linux:** Instala `espeak-ng` primero:
  ```bash
  sudo apt-get install espeak-ng
  pip install pyttsx3
  ```

---

## 🐛 Troubleshooting

### "No module named 'mediapipe'"
```bash
pip install mediapipe opencv-python pyttsx3
```

### "No se pudo abrir la cámara"
- Verifica que tu cámara funciona: `v4l2-ctl --list-devices` (Linux)
- Comprueba permisos: `sudo usermod -a -G video $USER`

### "No se oye la voz"
- Verifica el volumen del altavoz
- En Linux: `alsamixer` para ajustar niveles
- Instala `espeak-ng`: `sudo apt-get install espeak-ng`

### "Pantalla negra / Rendimiento lento"
- Reduce la resolución de la cámara en `app.py`
- Cierra otras aplicaciones
- En Raspberry Pi, actualiza a Pi 5 o usa lite

---

## 📱 Controles

| Tecla | Acción |
|-------|--------|
| `Q` | Salir (pantalla completa) |
| `ESC` | Salir (alternativo) |

---

## 🎯 Próximos Pasos

1. **Personaliza las caras:** Reemplaza los PNGs en `faces/` con tus propias imágenes
2. **Añade sonidos:** Coloca archivos `.wav` en `sounds/` (opcional)
3. **Despliega en Raspberry Pi:** Usa los scripts `setup.sh` y `start.sh`
4. **Integra con be-more-agent:** El proyecto es compatible con el framework be-more-agent

---

## 📄 Licencia

MIT License - Libre para usar y modificar

---

¿Preguntas? Revisa `app.py` o contacta al desarrollador.
