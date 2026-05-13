@echo off
REM ========================================================================
REM Saludo con Cámara - Start Script para Windows
REM ========================================================================

setlocal enabledelayedexpansion

REM Cambiar a la carpeta del script
cd /d "%~dp0" || exit /b 1

REM Activar entorno virtual
if exist venv (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
) else (
    echo No se encontró entorno virtual. Ejecuta setup.bat primero.
    exit /b 1
)

REM Verificar modelo
if not exist models\hand_landmarker.task (
    echo No se encontró models\hand_landmarker.task
    echo Ejecuta primero: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Ejecutar la aplicación
echo Iniciando Saludo con Cámara...
python app.py

pause
