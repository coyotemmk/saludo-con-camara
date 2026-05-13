# 🤖 Saludo con Cámara - Multiplataforma

**Aplicación de detección de saludos con cámara, animaciones y síntesis de voz.**

Diseñada para funcionar en **Windows**, **Linux** y **Raspberry Pi**.

---

## ⚡ Quick Start

### Windows
```powershell
# Instalar
pip install -r requirements.txt

# Ejecutar
python app.py
```

### Linux / Raspberry Pi
```bash
# Instalación automática
chmod +x setup.sh start.sh
./setup.sh

# Ejecutar
./start.sh
```

---

## 📋 Requisitos

- Python 3.8+
- Cámara USB conectada
- OpenCV, MediaPipe, pyttsx3

---

## 🎯 Características

✅ **Detección de saludos con manos** (MediaPipe)  
✅ **Síntesis de voz multiplataforma** (pyttsx3)  
✅ **Animaciones con PNGs** (bmo o tu personaje)  
✅ **Pantalla completa**  
✅ **Soporte de 2+ manos simultáneas**  
✅ **Ajuste de aspect ratio automático**  

---

## 🔧 Instalación Detallada

Ver [INSTALACION.md](INSTALACION.md)

---

## 🚀 Despliegue en Raspberry Pi

1. Clona o descarga el proyecto
2. Ejecuta `./setup.sh` (instala dependencias del sistema)
3. Ejecuta `./start.sh` para iniciar

### En Raspberry Pi OS Lite:
```bash
sudo apt-get install -y python3-pip python3-venv libatlas-base-dev
git clone <repo>
cd Saludo-con-camara
./setup.sh
./start.sh
```

---

## 📝 Uso

- **Levanta una mano** frente a la cámara
- **Muévela de lado a lado** para saludar
- La app dirá "Hola" 🎉
- **Presiona Q** para salir

---

## 🎨 Personalización

### Cambiar el personaje
Reemplaza los archivos `.png` en:
- `faces/listening/` → animación en reposo
- `faces/speaking/` → animación al hablar

### Cambiar el texto
En `app.py`, línea ~340:
```python
tts.say("Tu mensaje aquí")
```

### Ajustar parámetros
```python
cooldown_seconds = 3.0      # Espera entre saludos
listening_interval = 2.0    # Cambio de cara
speaking_interval = 0.15    # Velocidad al hablar
x_movement_threshold = 0.05 # Sensibilidad del vaivén
```

---

## 🛠️ Comandos Útiles

| Comando | Descripción |
|---------|-------------|
| `python app.py` | Ejecutar en Windows/Linux |
| `./start.sh` | Ejecutar en Linux (después de setup.sh) |
| `./setup.sh` | Installer en Linux (primeira vez) |
| `python -m venv venv` | Crear entorno virtual manual |
| `pip install -r requirements.txt` | Instalar dependencias |

---

## 🐛 Problemas Comunes

### "No module named 'mediapipe'"
```bash
pip install mediapipe opencv-python pyttsx3 numpy pillow
```

### Cámara no se abre
- Verifica: `v4l2-ctl --list-devices` (Linux)
- Dale permisos: `sudo usermod -a -G video $USER`
- Reinicia el terminal

### Sin voz en Linux
```bash
sudo apt-get install espeak espeak-ng
pip install pyttsx3
```

### Pantalla completa lenta
- Reduce resolución
- En Pi, cierra apps innecesarias
- Usa Raspberry Pi 5 si es posible

---

## 📦 Archivos Importantes

```
.
├── app.py              # Aplicación principal
├── setup.sh            # Installer para Linux
├── setup.bat           # Installer para Windows
├── start.sh            # Launcher para Linux
├── start.bat           # Launcher para Windows
├── requirements.txt    # Dependencias
├── INSTALACION.md      # Guía detallada
├── faces/              # Animaciones (.png)
├── models/             # Modelos IA (se descargan)
└── sounds/             # Sonidos (opcional)
```

---

## 🌍 Multiplataforma

| SO | Soportado | Script | TTS |
|-----|-----------|--------|-----|
| Windows | ✅ | `start.bat` | pyttsx3 |
| Linux | ✅ | `start.sh` | pyttsx3 + espeak |
| Raspberry Pi | ✅ | `start.sh` | pyttsx3 + espeak |
| macOS | ⚠️ | Manual | pyttsx3 |

---

## 📄 Licencia

MIT - Libre para usar y modificar

---

**¿Preguntas?** Revisa INSTALACION.md o el código en app.py

Happy coding! 🚀
