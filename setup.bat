@echo off
REM ========================================================================
REM Saludo con Cámara - Setup Script para Windows
REM ========================================================================

setlocal enabledelayedexpansion

echo.
echo =====================================
echo  🤖 Instalando Saludo con Camara
echo =====================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado o no está en el PATH
    echo.
    echo Descarga Python desde: https://www.python.org/downloads/
    echo IMPORTANTE: Marca "Add Python to PATH" durante la instalación
    pause
    exit /b 1
)

echo ✓ Python encontrado

REM Crear entorno virtual
echo.
echo 🔧 Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo ❌ Error: No se pudo crear el entorno virtual
    pause
    exit /b 1
)

echo ✓ Entorno virtual creado

REM Activar entorno virtual
echo.
echo 🔌 Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Error: No se pudo activar el entorno virtual
    pause
    exit /b 1
)

REM Actualizar pip
echo.
echo 📦 Actualizando pip...
python -m pip install --upgrade pip --quiet

REM Instalar dependencias desde requirements.txt
echo 📦 Instalando dependencias (OpenCV, MediaPipe, pyttsx3)...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ❌ Error: No se pudieron instalar las dependencias
    echo Intenta manualmente: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Instalar Piper TTS para voces de alta calidad
echo 📦 Instalando Piper TTS para voces de calidad...
pip install piper-tts --quiet
if errorlevel 1 (
    echo ⚠️  Advertencia: No se pudo instalar Piper TTS
    echo La app funcionará con pyttsx3 (voz del sistema)
)

REM Verificar instalaciones
echo.
echo ✓ Verificando instalaciones...
python -c "import cv2; import mediapipe; import pyttsx3; print('✅ Todas las dependencias instaladas correctamente')"
if errorlevel 1 (
    echo ❌ Error: Las dependencias no se instalaron correctamente
    pause
    exit /b 1
)

REM Crear carpeta de modelos si no existe
if not exist "models" mkdir models

REM Descargar modelo de Hand Landmarker si no existe
if not exist "models\hand_landmarker.task" (
    echo.
    echo ⏳ Descargando modelo de Hand Landmarker (puede tardar 1-2 min)...
    powershell -Command "& {(New-Object System.Net.WebClient).DownloadFile('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'models\hand_landmarker.task')}"
    if errorlevel 1 (
        echo ⚠️  Advertencia: No se pudo descargar el modelo
        echo Se descargará automáticamente cuando ejecutes la app
    )
)

REM Éxito
echo.
echo =====================================
echo ✅ ¡Instalación completada!
echo =====================================
echo.
echo 📝 Para ejecutar la app:
echo    1. Abre PowerShell o CMD en esta carpeta
echo    2. Ejecuta: python app.py
echo.
echo    O haz doble click en: start.bat
echo.
echo 🎯 Carpetas importantes:
echo    - faces/        → Animaciones del personaje
echo    - models/       → Modelos de IA
echo    - sounds/       → Sonidos opcionales
echo    - voice/        → Modelos de voz Piper
echo.
echo 🔊 Para usar voces Piper en español:
echo    1. Edita config.json
echo    2. Cambia "tts_backend" a "piper"
echo    3. Ejecuta: python app.py
echo.
pause
