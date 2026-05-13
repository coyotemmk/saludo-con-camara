#!/bin/bash

# ========================================================================
# Saludo con Cámara - Setup Script para Linux/Raspberry Pi
# ========================================================================

echo "🤖 Instalando Saludo con Cámara..."

# Detectar el sistema operativo
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Sistema: Linux"
    
    # Actualizar lista de paquetes
    echo "📦 Actualizando paquetes del sistema..."
    sudo apt-get update
    
    # Instalar dependencias del sistema
    echo "📚 Instalando dependencias del sistema..."
    sudo apt-get install -y \
        python3 python3-pip python3-venv \
        libatlas-base-dev libjasper-dev libtiff5 libjasper1 libharfbuzz0b libwebp6 libtiff5 \
        libopenjp2-7 libopenjp2-7-dev librpi-gpio-python3-dev \
        libsrtp2-1 libopenjp2-7 \
        python3-opencv libopencv-dev \
        python3-pip python3-dev \
        alsa-utils libsndfile1 \
        espeak espeak-ng
    
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "✓ Sistema: Windows (PowerShell/Git Bash)"
    echo "⚠️  En Windows, ejecuta directamente: python -m venv venv && .\\venv\\Scripts\\Activate && pip install -r requirements.txt"
else
    echo "⚠️  Sistema no completamente soportado: $OSTYPE"
fi

# Crear entorno virtual
echo "🔧 Creando entorno virtual..."
python3 -m venv venv

# Resolver binarios del entorno virtual (sin confiar en activate)
VENV_PYTHON="./venv/bin/python"
VENV_PIP="./venv/bin/pip"

# Verificar que se creó bien
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: No se pudo crear el entorno virtual."
    echo "Intenta: python3 -m venv venv"
    exit 1
fi

echo "✓ Entorno virtual creado en: $(pwd)/venv"

# Actualizar pip (DENTRO del entorno virtual)
echo "📦 Actualizando pip..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

# Instalar dependencias Python (DENTRO del entorno virtual)
echo "📦 Instalando OpenCV, MediaPipe, pyttsx3..."
"$VENV_PYTHON" -m pip install -r requirements.txt --quiet

# Verificar que se instalaron
echo "✓ Verificando instalaciones..."
"$VENV_PYTHON" -c "import cv2; import mediapipe; import pyttsx3; print('✅ Todas las dependencias instaladas correctamente')" || {
    echo "❌ Error: Las dependencias no se instalaron correctamente"
    echo "Intenta manualmente:"
    echo "  $VENV_PYTHON -m pip install -r requirements.txt"
    exit 1
}

# Descargar modelo de MediaPipe (Hand Landmarker)
echo "⏳ Descargando modelo de Hand Landmarker (esto puede tardar)..."
mkdir -p models
if [ ! -f models/hand_landmarker.task ]; then
    echo "Descargando hand_landmarker.task..."
    wget -O models/hand_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

echo ""
echo "✅ ¡Instalación completada!"
echo ""
echo "📝 Para ejecutar la app:"
echo "   ./start.sh   (en Linux/Raspberry Pi)"
echo "   python app.py   (en Windows)"
echo ""
echo "🎯 Carpetas importantes:"
echo "   - faces/        → Coloca los PNGs de animaciones aquí"
echo "   - models/       → Modelos de IA (Hand Landmarker)"
echo "   - sounds/       → Sonidos para la app"
